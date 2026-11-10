from dataclasses import dataclass, field
import ast
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from SPARQLWrapper import SPARQLWrapper
import requests
import fnmatch
from typing import Iterable, Optional, Dict, Any, Tuple, List
import warnings

from .mesh_tools import iter_servers_with_services_matching as _iter
from .mesh_tools import _canonize_string

@dataclass
class SparqlQuery:
    """
    query:   the SPARQL query text
    qvars:    expected variable order (without '?'), or None to infer from results
    headers: display headers (same length as vars); if None, fall back to vars
    """
    query: str = ""
    qvars: list[str] | None = None
    headers: list[str] | None = None

    def resolved_headers(self) -> list[str]:
        if self.headers:
            return self.headers
        return self.qvars or []

def infer_vars_from_results(result_json: dict) -> list[str]:
    '''
    Infers "headers/query-variables from SPARQL results JSON of the standard form:
    {"head": {"vars": [...]}, "results": {...}}
    '''
    return result_json.get("head", {}).get("vars", [])

def rectify_endpoints(endpoint):
    ep_1 = endpoint.replace(":None/api/jena", "/api/v1/jena")
    ep_2 = ep_1.replace(":443/api/jena", "/api/v1/jena")
    return ast.literal_eval(ep_2)

def make_dataframe(result, columns):
    liste = []
    for r in result['results']['bindings']:
        row = []
        for k in r.keys():
            row.append(r[k]['value'])
        liste.append(row)
    df = pd.DataFrame(liste, columns = columns)
    return df

def list_sparql_endpoints(
    partners: "RecursiveNamespace",
    service_patterns: Iterable[str] = ("*ontodocker*",),
    *,
    verify: Optional[bool | str] = True,
    timeout: Tuple[float, float] = (5.0, 5.0),
    scheme: str = "https",
    case_insensitive: bool = True,
    print_to_screen: bool = True,
) -> Dict[str, List[str]]:
    """
    Enumerate SPARQL endpoints for services matching `service_patterns`.

    Args:
      partners: Root RecursiveNamespace (company → server → services).
      service_patterns: Shell-style patterns for **sanitized** service keys
                       (e.g. ["*ontodocker*"]).
      verify: requests' TLS verification (True/False or CA bundle path).
      timeout: (connect_timeout, read_timeout) for HTTP GETs.
      scheme: "http" or "https" to prefix when `service.address` has no scheme.
      case_insensitive: Case-insensitive matching for `service_patterns`.
      print_to_screen: If True, prints per-service summaries.

    Returns:
      Dict mapping "<Company>.<Server>" (sanitized) → list of endpoint URLs.
      Services that error are skipped with a warning.
    """
    out: Dict[str, List[str]] = {}

    for company, server, svc_key, service in _iter(
        partners, service_patterns, case_insensitive=case_insensitive
    ):
        address = getattr(service, "address", None)
        token = getattr(service, "token", None)
        if not address:
            continue

        # Build base URL (respect existing scheme if present)
        base = address if address.startswith(("http://", "https://")) else f"{scheme}://{address}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        try:
            resp = requests.get(f"{base}/api/v1/endpoints", headers=headers, timeout=timeout, verify=verify)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"[list_sparql_endpoints] Connection error for {base}: {e!s}", category=UserWarning)
            continue
        except requests.exceptions.RequestException as e:
            warnings.warn(f"[list_sparql_endpoints] HTTP error for {base}: {e!s}", category=UserWarning)
            continue

        # The endpoint service may return a Python-literal-ish list; reuse existing helper
        try:
            endpoints = rectify_endpoints(resp.content.decode())
            if not isinstance(endpoints, list):
                raise TypeError("endpoint list is not a JSON/list literal")
        except Exception as e:
            warnings.warn(f"[list_sparql_endpoints] Could not parse endpoints from {base}: {e!s}", category=UserWarning)
            continue

        key = _canonize_string(f"{company}.{server}")
        out[key] = [str(ep) for ep in endpoints]

        if print_to_screen:
            print(company)
            print(f'Available SPARQL-endpoints at "{address}":')
            for ep in out[key]:
                print(ep)
            print("")

    return out

def send_query(endpoint: str | None = None,
               token: str | None = None,
               query: str | None = None,
               columns: list[str] | None = None,
               print_to_screen: bool = True
              ) -> pd.DataFrame:
    """
    Send a SPARQL query to a provided endpoint.
    Convert the result to a pandas dataframe and apply provided column headers.
    Optionally print some context to screen.
    Return the dataframe.

    Args:

    Returns:
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat('json')
    sparql.addCustomHttpHeader("Authorization", f'Bearer {token}')
    sparql.setQuery(query)
    result = sparql.queryAndConvert()
    result_df = make_dataframe(result, columns)
    if print_to_screen:
        print(f'Sending query to "{endpoint}". Result:')
        print(result_df)
        print("")
    return result_df

def query_instance(
    name: str,
    address: str,
    token: Optional[str],
    datasets: Iterable[str] = ("",),      # shell-style patterns for dataset names; "" → no filter
    query: Optional[str] = None,          # SPARQL query text
    columns: Optional[list[str]] = None,  # DataFrame column headers for the result
    print_to_screen: bool = True          # print endpoint + DataFrame after each query
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Query all dataset endpoints of a single service, optionally filtered by dataset patterns.

    Args:
      name: Logical service name (sanitized into a key).
      address: Hostname/FQDN of the service; scheme is added if missing.
      token: Bearer token for Authorization header (None → no header).
      datasets: Shell-style patterns against the dataset name (penultimate URL segment),
                e.g. ('damask*', 'public'); ('',) means no filtering.
      query: SPARQL query string.
      columns: DataFrame column names for the result.
      print_to_screen: Whether to print a short message plus the DataFrame per dataset.

    Returns:
      {sanitized_service_name: {dataset_name: {"endpoint","query","result"}}}
      where "result" is a pandas.DataFrame. If the service is unreachable, the returned
      dict contains the sanitized service name with an empty mapping.
    """
    import warnings

    # Ensure scheme for HTTP call
    base = address if address.startswith(("http://", "https://")) else f"http://{address}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sname = _canonize_string(name)
    out.setdefault(sname, {})

    # Try to fetch the endpoint list; warn on connection errors and return empty mapping
    try:
        resp = requests.get(f"{base}/api/v1/endpoints", headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        warnings.warn(f"[query_instance] Connection error for {base}: {e!s}", category=UserWarning)
        return out
    except requests.exceptions.RequestException as e:
        warnings.warn(f"[query_instance] HTTP error for {base}: {e!s}", category=UserWarning)
        return out

    endpoints = rectify_endpoints(resp.content.decode())

    # Prepare shell-style dataset matcher (case-insensitive)
    pats = tuple((p or "").casefold() for p in (datasets or ("",)))
    def dataset_matches(ds_name: str) -> bool:
        if pats == ("",):
            return True
        n = ds_name.casefold()
        return any(fnmatch.fnmatchcase(n, pat) for pat in pats)

    for ep in endpoints:
        parts = ep.rstrip("/").split("/")
        ds_name = parts[-2] if len(parts) >= 2 else parts[-1]
        if not dataset_matches(ds_name):
            continue

        try:
            df: pd.DataFrame = send_query(
                endpoint=ep,
                token=token,
                query=query,
                columns=columns,
                print_to_screen=print_to_screen,
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"[query_instance] Connection error querying {ep}: {e!s}", category=UserWarning)
            continue
        except Exception as e:
            # Catch other transport/driver errors (e.g., from SPARQLWrapper) without stopping the loop
            warnings.warn(f"[query_instance] Query failed for {ep}: {e.__class__.__name__}: {e!s}", category=UserWarning)
            continue

        out[sname].setdefault(ds_name, {})
        out[sname][ds_name]["endpoint"] = ep
        out[sname][ds_name]["query"] = query
        out[sname][ds_name]["result"] = df

    return out


def federated_query(
    partners: "RecursiveNamespace",
    query: str,
    columns: list[str],
    print_to_screen: bool = True,
    *,
    datasets: Iterable[str] = ("",),
    service_patterns: Iterable[str] = ("*ontodocker*",),  # shell-style service keys
    case_insensitive: bool = True,                                     # match services case-insensitively
    first_match_per_server: bool = True                                 # stop after first matching service on a server
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Traverse partners (company → server → services) using the iterator and run the query
    on services whose sanitized names match `service_patterns`.

    Args:
      partners: Root RecursiveNamespace (company → server → services hierarchy).
      datasets: Shell-style patterns for dataset names (passed to `query_instance`).
      query: SPARQL query string.
      columns: Expected DataFrame column names for results.
      print_to_screen: Whether each `query_instance` prints its DataFrame.
      service_patterns: Shell-style patterns matched against sanitized service keys
                        (e.g., ('ontodocker', 'ontodocker_*')).
      case_insensitive: Case-insensitive service name matching if True.
      first_match_per_server: If True, run the query only for the first matching service per server.

    Returns:
      Dict keyed by "company.server" (sanitized):
        {"Company_Server": { service_name: { dataset_name: {"endpoint","query","result"} } }}
    """
    import warnings
    from .mesh_tools import iter_servers_with_services_matching as _iter

    # detect if datasets is set to default
    ds_tuple = tuple(datasets or ("",))
    no_ds_filter = (len(ds_tuple) == 1 and (ds_tuple[0] or "") == "")
    
    seen_servers: set[tuple[str, str]] = set()
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for company_key, server_key, svc_key, svc in _iter(
        partners, service_patterns, case_insensitive=case_insensitive
    ):
        srv_id = (company_key, server_key)
        if first_match_per_server and srv_id in seen_servers:
            continue

        name = getattr(svc, "name", svc_key)
        address = getattr(svc, "address", None)
        token = getattr(svc, "token", None)
        if not address:
            continue

        agg_key = _canonize_string(f"{company_key}.{server_key}")

        with warnings.catch_warnings():
            # maybe this ist not the best was to do this
            if not no_ds_filter:
                warnings.simplefilter("ignore", category=UserWarning)

            try:
                results[agg_key] = query_instance(
                    name=name,
                    address=address,
                    token=token,
                    datasets=ds_tuple,
                    query=query,
                    columns=columns,
                    print_to_screen=print_to_screen,
                )
            except requests.exceptions.ConnectionError as e:
                # This warning is also suppressed when not no_ds_filter
                warnings.warn(f"[federated_query] Connection error for {address}: {e!s}", category=UserWarning)
                continue

        seen_servers.add(srv_id)

    return results