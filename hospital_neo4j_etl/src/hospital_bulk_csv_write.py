import os
import logging
from retry import retry
from neo4j import GraphDatabase

HOSPITALS_CSV_PATH = os.getenv("HOSPITALS_CSV_PATH")
PAYERS_CSV_PATH = os.getenv("PAYERS_CSV_PATH")
PHYSICIANS_CSV_PATH = os.getenv("PHYSICIANS_CSV_PATH")
PATIENTS_CSV_PATH = os.getenv("PATIENTS_CSV_PATH")
VISITS_CSV_PATH = os.getenv("VISITS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    )

LOGGER = logging.getLogger(__name__)

NODES = ["Hospital", "Payer", "Physician", "Patient", "Visit", "Review"]


#creates and runs queries enforcing each node to have a unique ID
def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node}) REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


#instantiate a driver that connects Neo4j instance and set uniqueness constraints for each hospital system node.
@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)


    LOGGER.info("Loading hospital nodes")
    with driver.session(database = "neo4j") as session:
        """
        A Cypher query to load data from a CSV file specified by HOSPITALS_CSV_PATH.
        LOAD CSV WITH HEADERS: Specifies that the CSV has headers.
        MERGE: Ensures that nodes with the specified properties (id, name, state_name) are created if they don't already exist, 
        avoiding duplicates.
        h:Hospital -> (represent a node in neo4j as Hospital)"""

        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{HOSPITALS_CSV_PATH}' AS hospitals
        MERGE (h:Hospital {{id: toInteger(hospitals.hospital_id),
                            name: hospitals.hospital_name,
                            state_name: hospitals.hospital_state}});
        """
        _ = session.run(query, {})
    

    LOGGER.info("Loading payer nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PAYERS_CSV_PATH}' AS payers
        MERGE (p:Payer {{id: toInteger(payers.payer_id),
        name: payers.payer_name}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading physician nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PHYSICIANS_CSV_PATH}' AS physicians
        MERGE (p:Physician {{id: toInteger(physicians.physician_id),
                            name: physicians.physician_name,
                            dob: physicians.physician_dob,
                            grad_year: physicians.physician_grad_year,
                            school: physicians.medical_school,
                            salary: toFloat(physicians.salary)
                            }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading visit nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
        MERGE (v:Visit {{id: toInteger(visits.visit_id),
                            room_number: toInteger(visits.room_number),
                            admission_type: visits.admission_type,
                            admission_date: visits.date_of_admission,
                            test_results: visits.test_results,
                            status: visits.visit_status
        }})
            ON CREATE SET v.chief_complaint = visits.chief_complaint
            ON MATCH SET v.chief_complaint = visits.chief_complaint
            ON CREATE SET v.treatment_description = visits.treatment_description
            ON MATCH SET v.treatment_description = visits.treatment_description
            ON CREATE SET v.diagnosis = visits.primary_diagnosis
            ON MATCH SET v.diagnosis = visits.primary_diagnosis
            ON CREATE SET v.discharge_date = visits.discharge_date
            ON MATCH SET v.discharge_date = visits.discharge_date
         """
        _ = session.run(query, {})

    LOGGER.info("Loading patient nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PATIENTS_CSV_PATH}' AS patients
        MERGE (p:Patient {{id: toInteger(patients.patient_id),
                        name: patients.patient_name,
                        sex: patients.patient_sex,
                        dob: patients.patient_dob,
                        blood_type: patients.patient_blood_type
                        }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading review nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{REVIEWS_CSV_PATH}' AS reviews
        MERGE (r:Review {{id: toInteger(reviews.review_id),
                         text: reviews.review,
                         patient_name: reviews.patient_name,
                         physician_name: reviews.physician_name,
                         hospital_name: reviews.hospital_name
                        }});
        """
        _ = session.run(query, {})