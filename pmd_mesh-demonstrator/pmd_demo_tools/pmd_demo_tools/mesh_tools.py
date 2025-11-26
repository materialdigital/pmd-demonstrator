from __future__ import annotations
from typing import Callable, Iterable, Mapping, Any, Optional
import copy
from types import SimpleNamespace
import re
import requests
import fnmatch
import warnings
import unicodedata
import sys
import json

def _canonize_string(name: str) -> str:
    """
    Canonize a string 'name' to match requirements for Python identifiers:
    - Everything NOT a letter, number or underscore is substituted with an underscore.
    - A leading digit prepended with the string "num"
    - german umlaute are replaced ba theri 2-vowel representation
    - some strings are treated specially (&)
    """
    # Normalize so composed/decomposed forms behave the same
    s = unicodedata.normalize("NFC", name)
    s = s.translate({
        ord('ä'): 'ae',
        ord('Ä'): 'Ae',
        ord('ö'): 'oe',
        ord('Ö'): 'Oe',
        ord('ü'): 'ue',
        ord('Ü'): 'Ue',
        ord('ß'): 'ss',
        ord('&'): '_and_',
        ord('+'): '_plus_',
    })
    
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s) # replace anything NOT a letter, number or underscore by underscore

    s = re.sub(r'_+', '_', s).strip('_') #collapse multiple underscores
    
    if s and s[0].isdigit():
        s = 'num_' + s # prepend 'num' if 'name' starts with a digit
    
    return s

def _sanitize_keys_unique(d: dict, sanitizer=_canonize_string) -> dict:
    """
    Sanitize keys using a passed sanitizer.
    Check key uniqueness: make sure that sanitized keys do not collapse onto one another.
    """
    out = {}
    seen = {}  # sanitized -> original (first seen)
    for k, v in d.items():
        sk = sanitizer(k)
        if sk in out:
            # collision: different keys on the same "level" map on the same sanitized string
            prev = seen[sk]
            raise ValueError(f"Key collision after sanitization: {k!r} and {prev!r} → {sk!r}")
        out[sk] = v
        seen[sk] = k
    return out

class RecursiveNamespace(SimpleNamespace):
    """
    RecusiveNamespace - just like its super class - maps its attributes ("keys") to its __dict__.
    In contrast to SimpleNamespace, this is done recursively for nested dicts,
    making their content dot-accessible.
    """
    @classmethod
    def _map_entry(cls, entry):
        """
        Recursively walk through dicts and map their keys to cls.__dict__
        Lists of dicts are also handeled, to convert these fully
        Recusion is stopped, when arriving at primitives.
        """
        if isinstance(entry, dict):
            sanitized = _sanitize_keys_unique(entry, sanitizer=_canonize_string)
            return cls(**{k: cls._map_entry(v) for k, v in sanitized.items()})
        if isinstance(entry, list):
            return [cls._map_entry(v) for v in entry]
        return entry

    @classmethod
    def _to_kwargs(cls, mapping: dict) -> dict:
        sanitized = _sanitize_keys_unique(mapping, sanitizer=_canonize_string)
        return {k: cls._map_entry(v) for k, v in sanitized.items()}

    def __init__(self, _processed: bool = False, **kwargs):
        # Only preprocess if caller passed raw data
        if not _processed:
            kwargs = self.__class__._to_kwargs(kwargs)
        super().__init__(**kwargs)

    @classmethod
    def from_mapping(cls, mapping: dict):
        """Public constructor for raw dicts (e.g., JSON objects)."""
        return cls(_processed=True, **cls._to_kwargs(mapping))

    def set(self, key: str, value, *, overwrite: bool = True) -> None:
        """Attach a single key/value at this node, with sanitization + recursive mapping."""
        sk = _canonize_string(key)
        if not overwrite and hasattr(self, sk):
            raise KeyError(f"Key already exists: {sk!r}")
        mapped = self.__class__._map_entry(value)
        setattr(self, sk, mapped)

    def update_ns(self, mapping: dict, *, overwrite: bool = True) -> None:
        """Bulk attach keys from a dict; sanitized + recursively mapped."""
        clean = _sanitize_keys_unique(mapping, sanitizer=_canonize_string)
        for k, v in clean.items():
            self.set(k, v, overwrite=overwrite)

    def ensure_path(self, *path: str) -> "RecursiveNamespace":
        """Create/return a nested path of nodes (sanitized keys)."""
        node = self
        for part in path:
            sk = _canonize_string(part)
            if not hasattr(node, sk) or not isinstance(getattr(node, sk), RecursiveNamespace):
                setattr(node, sk, self.__class__(_processed=True))  # empty namespace
            node = getattr(node, sk)
        return node
    
    def show(
        self,
        skip_keys: set[str] | list[str] = {"token"},
        skip_string: str = "<SECRET>",
        *,
        indent: int = 4,
        sort_keys: bool = True,
        stream=None,
        max_list: int = 100,
    ) -> None:
        """
        Print a readable view of the namespace.
        - skip_keys: keys (at any depth) whose values are replaced by skip_string
                     (match sanitized key names)
        - skip_string: replacement text
        - indent: spaces per nesting level
        - sort_keys: sort keys at each level for stable output
        - stream: file-like to write to (default: sys.stdout)
        - max_list: print at most this many list items per list
        """
        out = stream or sys.stdout
        redacted = set(skip_keys or ())

        def _w(line: str) -> None:
            out.write(line + "\n")

        def _pp(val, lvl: int, key_name: str | None = None):
            pad = " " * (indent * lvl)
            if key_name is not None:
                # key header
                if key_name in redacted:
                    _w(f"{pad}{key_name}: {skip_string}")
                    return

            if isinstance(val, RecursiveNamespace):
                if key_name is not None:
                    _w(f"{pad}{key_name}:")
                # For the root (key_name is None), print no header line
                items = vars(val).items()
                if sort_keys:
                    items = sorted(items, key=lambda kv: kv[0])
                for k, v in items:
                    _pp(v, lvl + (1 if key_name is not None else 0), k if key_name is not None else k)
            elif isinstance(val, list):
                label = f"{key_name}:" if key_name is not None else "-"
                _w(f"{pad}{label}")
                n = 0
                for i, item in enumerate(val):
                    if n >= max_list:
                        _w(f"{pad}{' ' * indent}… ({len(val) - max_list} more)")
                        break
                    # list entries get a dash; recurse if they are namespaces
                    if isinstance(item, RecursiveNamespace):
                        _w(f"{pad}{' ' * indent}-")
                        _pp(item, lvl + 2, None)
                    else:
                        _w(f"{pad}{' ' * indent}- {repr(item)}")
                    n += 1
            else:
                # primitives and other objects
                if key_name is not None:
                    _w(f"{pad}{key_name}: {repr(val)}")
                else:
                    _w(f"{pad}{repr(val)}")

        _pp(self, 0, None)

    def to_string(self, **kw) -> str:
        buf = io.StringIO()
        self.show(stream=buf, **kw)
        return buf.getvalue()

    def _to_plain(self, *, skip_keys=frozenset(), skip_string="<SECRET>"):
        """Return a JSON-safe nested structure (dict/list/primitives)."""
        def conv(val):
            if isinstance(val, RecursiveNamespace):
                return {k: conv(v) if k not in skip_keys else skip_string
                        for k, v in vars(val).items()}
            if isinstance(val, list):
                return [conv(x) for x in val]
            if isinstance(val, tuple) or isinstance(val, set):
                return [conv(x) for x in val]
            return val  # primitives or json-serializable objects
        return {k: conv(v) if k not in skip_keys else skip_string
                for k, v in vars(self).items()}

    def dump_json(
        self,
        path: str,
        *,
        skip_keys: set[str] | list[str] = {"token"},
        skip_string: str = "<SECRET>",
        indent: int = 2,
        sort_keys: bool = True,
        ensure_ascii: bool = False,
    ) -> None:
        """
        Write the namespace to a JSON file.
        - skip_keys: redact these key names at any depth by writing skip_string
        - Non-JSON types fall back to str() via json’s default
        """
        plain = self._to_plain(skip_keys=set(skip_keys), skip_string=skip_string)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                plain,
                f,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
                default=str,  # last-resort for odd values
            )

def namespace_object_hook(ns_cls=RecursiveNamespace):
    """
    Object hook for classes derived from RecusiveNamespace.
    The from_mapping() method must be overloaded.
    """
    def hook(d: dict):
        return ns_cls.from_mapping(d)
    return hook

def recursive_namespace_iterator(obj):
    for key, value in vars(obj).items():
        yield key, value  # recent level
        # Recursion for nested objects
        if isinstance(value, RecursiveNamespace):
            yield from recursive_iter(value)
        # handle lists which already contain RecursiveNamespace objects
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, RecursiveNamespace):
                    yield from recursive_iter(item)

def select_toplevel(rns: RecursiveNamespace, selection: Iterable[str], *, deepcopy=False):
    '''
    deepcopy=True makes a true, independent instance with its own state.
    deepcopy=False amkes a shallow copy, where the underlying dicts/rcn's are shared between the original instance and the "selection"
    '''
    # Guard: must be an instance, not the class, not a function, not a module
    if not isinstance(rns, RecursiveNamespace):
        raise TypeError(f"select_toplevel expected RecursiveNamespace instance, got {type(rns).__name__}")

    cls = rns.__class__  # preserve subclass type
    base = vars(rns)

    picked = {}
    for k in selection:
        if k not in base:
            raise KeyError(f"Key {k!r} not found at top level")
        v = base[k]
        picked[k] = copy.deepcopy(v) if deepcopy else v

    return cls(_processed=True, **picked)

def select_by_services(
    root: RecursiveNamespace,
    patterns: Iterable[str],
    *,
    case_insensitive: bool = True,
    deepcopy: bool = False,
    keep_empty_servers: bool = False,         # keep servers with no services/matches
    keep_all_services_on_match: bool = True,  # True: keep all services once any matches
) -> RecursiveNamespace:
    """
    Return a new RecursiveNamespace filtered by service names.

    - patterns: shell-style patterns matched against sanitized service keys
      (e.g., ["ontodocker", "ontodocker_*"])
    - keep_all_services_on_match=True keeps every service under a server that
      has at least one matching service; if False, only the matching services are kept.
    """
    pats = [p.casefold() for p in patterns] if case_insensitive else list(patterns)

    def matches(name: str) -> bool:
        n = name.casefold() if case_insensitive else name
        return any(fnmatch.fnmatchcase(n, p) for p in pats)

    new_companies = {}

    for company_name, company in vars(root).items():
        if not isinstance(company, RecursiveNamespace):
            continue

        new_servers = {}

        for server_name, server in vars(company).items():
            if not isinstance(server, RecursiveNamespace):
                continue

            svcs = getattr(server, "services", None)

            # Decide whether this server should be kept
            if isinstance(svcs, RecursiveNamespace):
                svc_items = list(vars(svcs).items())
                has_match = any(matches(svc_name) for svc_name, _ in svc_items)
            else:
                has_match = False

            if not has_match and not keep_empty_servers:
                continue  # drop this server

            # Build the server copy (or view)
            server_kwargs = {k: (copy.deepcopy(v) if deepcopy else v)
                             for k, v in vars(server).items() if k != "services"}
            server_new = server.__class__(_processed=True, **server_kwargs)

            if isinstance(svcs, RecursiveNamespace):
                if keep_all_services_on_match:
                    # keep ALL services under this server
                    services_payload = {k: (copy.deepcopy(v) if deepcopy else v)
                                        for k, v in svc_items}
                else:
                    # keep ONLY services that match
                    services_payload = {k: (copy.deepcopy(v) if deepcopy else v)
                                        for k, v in svc_items if matches(k)}
                services_new = svcs.__class__(_processed=True, **services_payload)
                setattr(server_new, "services", services_new)
            else:
                # No services Namespace (None or missing)
                if keep_empty_servers:
                    setattr(server_new, "services", None)

            new_servers[server_name] = server_new

        if new_servers:
            new_companies[company_name] = company.__class__(_processed=True, **new_servers)

    return root.__class__(_processed=True, **new_companies)

# PMD-server discovery
def fetch_mesh_listing(
    url: str = "http://mesh-listing.c.pmd.internal/api/v1/pmds",
    *,
    timeout: float = 10.0,
    verify: Optional[str | bool] = True,
    headers: Optional[Mapping[str, str]] = None,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """
    GET the mesh listing endpoint and return a list of dicts.
    - 'verify' accepts True/False or a CA bundle path.
    - Provide a 'session' if you want connection pooling/retries outside this helper.
    """
    s = session or requests
    resp = s.get(url, timeout=timeout, verify=verify, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list, got {type(data).__name__}")
    # (Optional) quick shape check
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} is not an object: {type(item).__name__}")
    return data

def index_mesh_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    key: str = "wg_mesh_dns_zone",
    key_func: Optional[Callable[[Mapping[str, Any]], str]] = None,
    sanitize_keys: bool = True,
    strict_unique: bool = True,
) -> dict[str, Any]:
    """
    Turn the list into a dict keyed by a chosen field (default: 'wg_mesh_dns_zone').
    - Use 'key_func' to compute keys dynamically (e.g., combine fields).
    - If 'sanitize_keys' is True, keys go through _canonize_string(...)
    - If 'strict_unique' is True, raise on duplicate keys; otherwise last one wins.
    """
    if key_func is None:
        def key_func(item: Mapping[str, Any]) -> str:
            try:
                return str(item[key])
            except KeyError as e:
                raise KeyError(f"Missing key {key!r} in entry: {item}") from e

    idx: dict[str, Any] = {}
    seen_raw: dict[str, str] = {}
    for item in entries:
        raw_k = key_func(item)
        k = _canonize_string(raw_k) if sanitize_keys else raw_k
        if strict_unique and k in idx:
            prev = seen_raw[k]
            raise ValueError(
                f"Duplicate index key after sanitization: {raw_k!r} vs {prev!r} -> {k!r}"
            )
        idx[k] = item
        seen_raw[k] = raw_k
    return idx

def mesh_listing_namespace(
    *,
    url: str = "http://mesh-listing.c.pmd.internal/api/v1/pmds",
    key: str = "wg_mesh_dns_zone",
    verify: Optional[str | bool] = True,
    timeout: float = 10.0,
) -> RecursiveNamespace:
    """
    Fetch → index → wrap into RecursiveNamespace where each top-level key is the chosen field.
    May result in collisison, when a key is the same for mutliple entries (e.g. when the same "company" hosts multiple servers).
    """
    entries = fetch_mesh_listing(url=url, timeout=timeout, verify=verify)
    idx = index_mesh_entries(entries, key=key, sanitize_keys=True, strict_unique=True)
    # Each value is still a plain dict with primitive fields/list-of-strings.
    # from_mapping will sanitize nested keys (none collide here) and give you dot-access.
    return RecursiveNamespace.from_mapping(idx)

def _trim_pmd_internal(label: str) -> str:
    """
    Remove trailing '.pmd.internal' in common separators:
      'kit-4.pmd.internal', 'kit-4-pmd-internal', 'kit_4_pmd_internal'
    """
    return re.sub(r'([._-])pmd([._-])internal$', '', label, flags=re.IGNORECASE)

def _group_by_company(
    entries: Iterable[Mapping[str, Any]],
    *,
    server_key: str = "wg_mesh_dns_zone",
    trim_pmd_internal: bool = False,
    server_label_func: Optional[callable] = None,
    strict_unique: bool = True,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """
    Build {company -> {server_label -> entry}}.

    - server_key: which field to use for the per-server key (default: wg_mesh_dns_zone)
    - trim_pmd_internal: if True, remove a trailing 'pmd.internal' (with ., -, _ separators)
    - server_label_func: optional callable(item) -> str to compute custom labels
    - strict_unique: raise if two servers under the same company collapse to the same label
    """
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}

    for it in entries:
        # top-level: company (sanitize later via from_mapping)
        company = it["company"]

        # compute raw server label
        if server_label_func is not None:
            raw_label = server_label_func(it)
        else:
            raw_label = str(it[server_key])

        if trim_pmd_internal:
            raw_label = _trim_pmd_internal(raw_label)

        # Put into nested dict; collisions handled here (before sanitization)
        bucket = grouped.setdefault(company, {})
        if strict_unique and raw_label in bucket:
            raise ValueError(
                f"Duplicate server label under company {company!r}: {raw_label!r}"
            )
        bucket[raw_label] = it

    return grouped

def mesh_namespace_grouped_by_company(
    *,
    server_key: str = "wg_mesh_dns_zone",
    trim_pmd_internal: bool = True,
    verify: bool | str | None = True,
    timeout: float = 10.0,
) -> RecursiveNamespace:
    entries = fetch_mesh_listing(timeout=timeout, verify=verify)
    grouped = _group_by_company(
        entries,
        server_key=server_key,
        trim_pmd_internal=trim_pmd_internal,
        strict_unique=True,
    )
    # Keys get sanitized and nested dicts mapped by your class:
    return RecursiveNamespace.from_mapping(grouped)

def fetch_zone_services(
    zone: str,
    base_url: str = "http://mesh-listing.c.pmd.internal/api/v1/pmds",
    *,
    timeout: float = 10.0,
    verify: Optional[str | bool] = True,
    session: Optional[requests.Session] = None,
) -> list[str]:
    """
    GET /api/v1/pmds/<zone> and return a list of FQDN (= fully qualified domain names; the full 'address') strings (services).
    """
    url = f"{base_url.rstrip('/')}/{zone}"
    s = session or requests
    r = s.get(url, timeout=timeout, verify=verify)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise TypeError(f"Expected a JSON list[str] for services at {url}")
    return data


DEFAULT_SERVICE_EXCLUDES = ["*app*", "jupyter-*", "ns", "*certbot*", "*nginx*", "wg", "uptime-kuma*", "mesh-listing", "ca"]

def _matches_any(name: str, patterns: Iterable[str], *, case_insensitive: bool = True) -> bool:
    if not patterns:
        return False
    n = name.casefold() if case_insensitive else name
    for p in patterns:
        pp = p.casefold() if case_insensitive else p
        if fnmatch.fnmatchcase(n, pp):
            return True
    return False

def services_mapping_from_hostnames(
    hosts: Iterable[str],
    *,
    exclude_patterns: Optional[Iterable[str]] = DEFAULT_SERVICE_EXCLUDES,
    case_insensitive: bool = True,
    raise_on_duplicate: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Build {short_service -> {name, address, token}} from a list of FQDNs,
    skipping services whose *short name* (leftmost label) matches any pattern.
    """
    services: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for fqdn in hosts:
        short = fqdn.split(".", 1)[0]  # e.g., "jupyter-mbruns-2ddev"
        if _matches_any(short, exclude_patterns or [], case_insensitive=case_insensitive):
            continue

        if raise_on_duplicate and short in seen:
            raise ValueError(f"Duplicate service name {short!r} derived from {fqdn!r}")
        seen.add(short)

        services[short] = {"name": short, "address": fqdn, "token": None}

    return services

def build_services_namespace_for_zone(
    zone: str,
    *,
    base_url: str = "http://mesh-listing.c.pmd.internal/api/v1/pmds",
    timeout: float = 10.0,
    verify: Optional[str | bool] = True,
    exclude_patterns: Optional[Iterable[str]] = DEFAULT_SERVICE_EXCLUDES,
) -> RecursiveNamespace:
    hosts = fetch_zone_services(zone, base_url=base_url, timeout=timeout, verify=verify)
    mapping = services_mapping_from_hostnames(hosts, exclude_patterns=exclude_patterns)
    return RecursiveNamespace.from_mapping(mapping)

def attach_services_in_place(
    partners_ns: RecursiveNamespace,
    *,
    base_url: str = "http://mesh-listing.c.pmd.internal/api/v1/pmds",
    timeout: float = 10.0,
    verify: Optional[str | bool] = True,
    on_error: str = "set_none",
    exclude_patterns: Optional[Iterable[str]] = DEFAULT_SERVICE_EXCLUDES,
) -> RecursiveNamespace:
    """
    Walk company→server and set server.services.
    On failure:
      - "propagate": raise the exception
      - "warn": warn and leave services unset
      - "skip": silently leave services unset
      - "set_none": set services = None
    """
    for company in vars(partners_ns).values():
        if not isinstance(company, RecursiveNamespace):
            continue
        for server in vars(company).values():
            if not isinstance(server, RecursiveNamespace):
                continue
            zone = getattr(server, "wg_mesh_dns_zone", None)
            if not zone:
                continue
            try:
                services_ns = build_services_namespace_for_zone(
                    zone,
                    base_url=base_url,
                    timeout=timeout,
                    verify=verify,
                    exclude_patterns=exclude_patterns,
                )
            except Exception as e:
                if on_error == "propagate":
                    raise
                if on_error == "warn":
                    warnings.warn(f"Failed fetching services for {zone}: {e}", category=UserWarning)
                elif on_error == "set_none":
                    setattr(server, "services", None)
                    warnings.warn(f"Failed fetching services for {zone}: {e}.", category=UserWarning)
                elif on_error == "skip":
                    pass  # explicitly do nothing
                else:
                    raise ValueError(f"Unknown on_error mode: {on_error!r}")
                continue
            else:
                setattr(server, "services", services_ns)
    return partners_ns

def iter_servers_with_services_matching(root, patterns, *, case_insensitive=True):
    pats = [p.casefold() for p in patterns] if case_insensitive else patterns
    def match(name): 
        n = name.casefold() if case_insensitive else name
        return any(fnmatch.fnmatchcase(n, p) for p in pats)

    for company_name, company in vars(root).items():
        if not isinstance(company, RecursiveNamespace):
            continue
        for server_name, server in vars(company).items():
            if not isinstance(server, RecursiveNamespace):
                continue
            svcs = getattr(server, "services", None)
            if not isinstance(svcs, RecursiveNamespace):
                continue
            for svc_name, svc in vars(svcs).items():
                if match(svc_name):
                    yield company_name, server_name, svc_name, svc

def attach_tokens_to_partners(
    partners: RecursiveNamespace,
    tokens_ns: RecursiveNamespace,
    *,
    overwrite: bool = True,
    warn_missing: bool = False,
) -> None:
    """
    Copy tokens from `tokens_ns` (company -> service -> token) into
    `partners` (company -> server -> services -> service -> token).

    Args:
      partners: company -> server -> services -> service namespaces.
      tokens_ns: company -> service namespaces holding `.token` fields.
                 (i.e., tokens.<Company>.<Service>.token)
      overwrite: if False, keep existing non-empty tokens in `partners`.
      warn_missing: warn when a matching token is not found.
    """
    for company_key, company in vars(partners).items():
        if not isinstance(company, RecursiveNamespace):
            continue

        # try to find the same company in the tokens tree
        tok_company = getattr(tokens_ns, company_key, None)
        if not isinstance(tok_company, RecursiveNamespace):
            if warn_missing:
                warnings.warn(f"[attach_tokens] no tokens for company {company_key}")
            continue

        for server_key, server in vars(company).items():
            if not isinstance(server, RecursiveNamespace):
                continue

            svcs = getattr(server, "services", None)
            if not isinstance(svcs, RecursiveNamespace):
                continue

            for svc_key, svc in vars(svcs).items():
                if not isinstance(svc, RecursiveNamespace):
                    continue

                tok_service = getattr(tok_company, svc_key, None)
                if not isinstance(tok_service, RecursiveNamespace):
                    if warn_missing:
                        warnings.warn(f"[attach_tokens] no token for {company_key}.{svc_key}")
                    continue

                token_val: Optional[str] = getattr(tok_service, "token", None)
                if token_val is None or token_val == "":
                    if warn_missing:
                        warnings.warn(f"[attach_tokens] empty token for {company_key}.{svc_key}")
                    continue

                if not overwrite:
                    existing = getattr(svc, "token", None)
                    if existing not in (None, ""):
                        continue

                setattr(svc, "token", token_val)