from pyiron_workflow import Workflow
import queries

@dataclass
class SparqlQuery:
    """Dataclass for SPARQL queries and column headers corresponding to the expected result."""
    query: str = ""
    columns: list[str] = field(default_factory=list)

@Workflow.wrap.as_function_node
def material_designation():
    """
    Define a query prompting for the value of materialDesignation.
    Return the query and a list of column headers corresponding to the expected result.
    """
    query = """
    PREFIX pmd: <https://w3id.org/pmd/co/>
    SELECT DISTINCT ?p ?matDesVal
    WHERE {
        ?s a pmd:TestPiece .
        ?p pmd:input ?s .
        ?p pmd:characteristic ?matDes .
        ?matDes a pmd:materialDesignation .
        ?matDes pmd:value ?matDesVal .
    }
    ORDER BY ?p
    """
    columns = ["URI", "materialDesignation"]
    return SparqlQuery(query, columns)