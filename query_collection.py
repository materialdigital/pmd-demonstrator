from dataclasses import dataclass


@dataclass
class SparqlQuery:
    """
    query:   the SPARQL query text
    query_vars: expected variable order without leading '?'.
    """

    query: str
    query_vars: list[str]


class TestQueries:
    """
    Queries for a simple dataset stored in "test_dataset.ttl.
    The queries and the corresponding query variables are defined.
    These queries are arbitrarily selected to fit the needs of the pmd-demonstrator.
    """

    @staticmethod
    def test_query() -> SparqlQuery:
        """
        Define a query on a simple test-dataset
        """
        query = """
        PREFIX ex:   <http://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?person (?personLabel AS ?person_label) ?project (?projectLabel AS ?project_label)
        WHERE {
          ?person a ex:Person ;
              ex:worksOn ?project ;
              rdfs:label ?personLabel .
          ?project rdfs:label ?projectLabel .
        }
        ORDER BY ?person
        """
        query_vars = ["person", "person_label", "project", "project_label"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def query_graph(limit: int | None = None) -> SparqlQuery:
        """
        Query all triples in the graph.
        """
        limit_clause = "" if limit is None else f"LIMIT {limit}"
        query = f"""
        SELECT (?s AS ?subject) (?p AS ?predicate) (?o AS ?object)
        WHERE {{ ?s ?p ?o }}
        {limit_clause}
        """.strip()
        query_vars = ["subject", "predicate", "object"]
        return SparqlQuery(query=query, query_vars=query_vars)


class S355queries:
    """
    Central class for all queries to the S355 dataset described using pmdco v2.07 and tto .
    The queries and the corresponding query variables are defined.
    These queries are arbitrarily selected to fit the needs of the pmd-demonstrator.
    """

    @staticmethod
    def material_designation():
        """
        Define a query prompting for the value of materialDesignation.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT DISTINCT (?p AS ?uri) (?matDesVal AS ?materialDesignation)
        WHERE {
            ?s a pmd:TestPiece .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?matDes .
            ?matDes a pmd:materialDesignation .
            ?matDes pmd:value ?matDesVal .
        }
        ORDER BY ?p
        """
        query_vars = ["uri", "materialDesignation"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def process_type(material: str = "S355"):
        """
        Define a query prompting for the value of processType.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT distinct (?p AS ?uri) (?type AS ?process_type)
        WHERE {{
            ?p a ?type .
            ?matDes a pmd:materialDesignation .
            ?matDes pmd:value "{material}"^^xsd:string .
            ?p pmd:characteristic ?matDes .
        }}
        ORDER BY ?p
        """.strip()
        query_vars = ["uri", "process_type"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def orientation():
        """
        Define a query prompting for the orientation in which a specimen
        was cut from the raw material relative to the rolling direction.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?rollingDirection AS ?cut_orientation)
        WHERE {
            ?s a pmd:TestPiece .
            ?p a pmd:TensileTest .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?characteristic .
            ?characteristic a pmd:MaterialRelated .
            ?characteristic pmd:value ?rollingDirection .
        }
        ORDER BY ?p
        """
        query_vars = ["uri", "cut_orientation"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def standard_and_extensiometer():
        """
        Define a query prompting for the applied standards/norms during the test.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?extensometerNameVal AS ?extensiometer_model) (?extensometerStandardVal AS ?standard)
        WHERE {
            ?s a pmd:TestPiece .
            ?p a pmd:TensileTest .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?metadata .
            ?extensometerName a pmd:NodeName .
            ?extensometerName pmd:value ?extensometerNameVal .
            ?extensometerStandard a pmd:Norm .
            ?extensometerStandard pmd:value ?extensometerStandardVal .
            FILTER (?extensometerName!=<https://w3id.org/pmd/ao/tte/_machineName>)
            FILTER (?extensometerStandard=<https://w3id.org/pmd/ao/tte/_extensometerStandard>)
        }
        ORDER BY ?p
        """
        query_vars = ["uri", "extensiometer_model", "standard"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def extensiometer():
        """
        Define a query prompting for the tensile test machines model name.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?extensometerNameVal AS ?Extensiometer_model)
        WHERE {
        ?s a pmd:TestPiece .
        ?p a pmd:TensileTest .
        ?p pmd:input ?s .
        ?p pmd:characteristic ?metadata .
        ?extensometerName a pmd:NodeName .
        ?extensometerName pmd:value ?extensometerNameVal .
        FILTER (?extensometerName!=<https://w3id.org/pmd/ao/tte/_machineName>)
        } ORDER BY ?p
        """
        query_vars = ["uri", "Extensiometer_model"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def specimen_id():
        """
        Define a query prompting for the specimen's ID.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?s AS ?specimen_id)
        WHERE {
        ?s a pmd:TestPiece .
        ?p a pmd:TensileTest .
        ?p pmd:input ?s .
        } ORDER BY ?p
        """
        query_vars = ["uri", "specimen_id"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def csv_url():
        """
        Define a query prompting for the URL pointing to the csv file containing strass-strain data.
        Return the query and a list of query variables corresponding to the expected result.
        """
        query = """
        PREFIX base: <https://w3id.org/pmd/co/>
        PREFIX csvw: <http://www.w3.org/ns/csvw#>
        SELECT (?p AS ?uri) ?url
        WHERE {
            ?p a base:TensileTest .
            ?p base:characteristic ?dataset .
            ?dataset a base:Dataset .
            ?dataset base:resource ?table .
            ?table a csvw:Table .
            ?table csvw:url ?url .
        }
        ORDER BY ?p
        """
        query_vars = ["uri", "url"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def primary_data(uri: str | None = None):
        """
        Define a query prompting for all PrimaryData values accossiated with a certain process, hinted via its URI.
        Return the query and a list of query variables corresponding to the expected result.

        Args:
            uri (str | None): URI used to identify the process of question
        Returns:
            ...
        """
        filter_clause = "" if uri is None else f'FILTER regex(str(?p), "{uri}")'
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?o AS ?quantity) (?v AS ?value) (?u AS ?unit)
        WHERE {{
            ?s a pmd:TestPiece .
            ?p a pmd:TensileTest .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?o .
            ?o a pmd:PrimaryData .
            ?o pmd:value ?v .
            ?o pmd:unit ?u .
            {filter_clause}
        }} ORDER BY ?p
        """.strip()
        query_vars = ["uri", "quantity", "value", "unit"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def secondary_data(uri: str | None = None):
        """
        Define a query prompting for all SecondaryData values accossiated with a certain process, hinted via its URI.
        Return the query and a list of query variables corresponding to the expected result.

        Args:
            uri (str | None): URI used to identify the process of question
        Returns:
            ...
        """
        filter_clause = "" if uri is None else f'FILTER regex(str(?p), "{uri}")'
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT distinct (?p AS ?uri) (?o AS ?quantity) (?v AS ?value) (?u AS ?unit)
        WHERE {{
            ?s a pmd:TestPiece .
            ?p a pmd:TensileTest .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?o .
            ?o a pmd:SecondaryData .
            ?o pmd:value ?v .
            ?o pmd:unit ?u .
            {filter_clause}
        }} ORDER BY ?p
        """.strip()
        query_vars = ["uri", "quantity", "value", "unit"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def metadata(uri: str | None = None):
        """
        Define a query prompting for all Metadata values accossiated with a certain process, hinted via its URI.
        Return the query and a list of query variables corresponding to the expected result.

        Args:
            uri (str | None): URI used to identify the process of question
        Returns:
            All properies described as pmd co metadata for all processes/ uris in the graph.
            If 'uri' is specified, only those for the related process are queried.
        """
        filter_clause = "" if uri is None else f'FILTER regex(str(?p), "{uri}")'
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/>
        SELECT DISTINCT (?p AS ?uri) (?o AS ?Quantity) (?v AS ?value) (?u AS ?unit)
        WHERE {{
            ?s a pmd:TestPiece .
            ?p a pmd:TensileTest .
            ?p pmd:input ?s .
            ?p pmd:characteristic ?o .
            ?o a pmd:Metadata .
            ?o pmd:value ?v .
            ?o pmd:unit ?u .
            {filter_clause}
        }} ORDER BY ?p
        """.strip()
        query_vars = ["uri", "Quantity", "value", "unit"]
        return SparqlQuery(query=query, query_vars=query_vars)

    @staticmethod
    def csv_columns(uri: str | None = None):
        """
        ...
        """
        filter_clause = "" if uri is None else f'FILTER (str(?p)="{uri}")'
        query = f"""
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX csvw: <http://www.w3.org/ns/csvw#>
        PREFIX base: <https://w3id.org/pmd/co/>
        
        SELECT (?p AS ?uri) ?url (?part AS ?name) (COUNT(DISTINCT ?mid) AS ?column_number) (?type AS ?quantity) ?unit
        WHERE {{
          # Relate test → dataset → table → url
          ?p a base:TensileTest ;
             base:characteristic ?dataset .
          ?dataset a base:Dataset ;
                   base:resource ?table .
          ?table a csvw:Table ;
                 csvw:url ?url ;
                 csvw:tableSchema ?schema .
        
          # Walk the column list and pick the current part
          ?schema csvw:column/rdf:rest* ?mid .
          ?mid rdf:rest* ?node .
          ?node rdf:first ?part .
        
          # Metadata for the part
          ?part a ?type ;
                base:unit ?unit .
        
          FILTER (?type != csvw:Column)
          {filter_clause}
        }}
        GROUP BY ?p ?url ?part ?type ?unit
        ORDER BY ?p ?column_number
        """
        query_vars = ["uri", "url", "name", "column_number", "quantity", "unit"]
        return SparqlQuery(query=query, query_vars=query_vars)
