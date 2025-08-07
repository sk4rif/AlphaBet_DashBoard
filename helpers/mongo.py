from pymongo import MongoClient, DESCENDING
import certifi

def push_data(database, sub_collection, data):
    # Step 1: Set up a client connection to MongoDB
    client = MongoClient(
        "mongodb+srv://CarlGrimaldi:AlphaBeta21$@alphabetcluster.x7lvc.mongodb.net/",
        tlsCAFile=certifi.where()
    )  # Use the appropriate URI if not local

    # Step 2: Access the database
    db = client[database]  # Replace "mydatabase" with your database name

    # Step 3: Access the collection
    collection = db[sub_collection]  # Replace "mycollection" with your collection name

    # Step 4: Insert a document (data) into the collection
    try:
        result = collection.insert_one(data)  # Attempt to insert a new document
        print("Data inserted with record id:", result.inserted_id)
    except Exception as e:
        # If an exception occurs (e.g., duplicate key error), update the document

        # Assuming `data` contains a unique identifier like `_id` or another key to find the document
        filter_criteria = {"_id": data["_id"]}  # Update this based on your unique key
        update_data = {"$set": data}  # Use `$set` to update fields

        try:
            update_result = collection.update_one(filter_criteria, update_data)
            if update_result.matched_count > 0:
                print("Data updated successfully.")
            else:
                print("No matching document found to update.")
        except Exception as update_error:
            print(f"Update failed: {update_error}")



def pull_data(database, sub_collection, _id):
    # Step 1: Set up a client connection to MongoDB
    client = MongoClient(
        "mongodb+srv://CarlGrimaldi:AlphaBeta21$@alphabetcluster.x7lvc.mongodb.net/",
        tlsCAFile=certifi.where()
    )  # Use the appropriate URI if not local

    # Step 2: Access the database
    db = client[database]  
    
    # Step 3: Access the collection
    collection = db[sub_collection]  
    
    if _id == 'latest':
        print("Retrieving the latest uploaded document...")
        # Retrieve the latest uploaded document (sorted by 'created_at' field)
        latest_data = collection.find_one(sort=[("_id", DESCENDING)])
        return latest_data
    
    elif _id is not None:
        # Retrieve a single document by _id
        data = collection.find_one({"_id": _id})  
        return data
    

    
    else:
        # Retrieve and return a list of all _id values in the collection
        return [doc["_id"] for doc in collection.find({}, {"_id": 1})]




def append_data(database, sub_collection, _id, data):
    """
    Appends values from a dictionary to the corresponding lists in a MongoDB document.
    If the document with _id does not exist, it inserts a new document using push_data.

    :param database: Name of the MongoDB database.
    :param sub_collection: Name of the collection.
    :param _id: The unique identifier (_id) of the document to update.
    :param data: A dictionary containing lists to append to the existing document.
    """

    # Step 1: Set up a client connection to MongoDB
    client = MongoClient(
        "mongodb+srv://CarlGrimaldi:AlphaBeta21$@alphabetcluster.x7lvc.mongodb.net/",
        tlsCAFile=certifi.where()
    )  # Use the appropriate URI if not local

    # Step 2: Access the database
    db = client[database]
    collection = db[sub_collection]

    # Step 3: Check if the document with _id exists
    existing_doc = collection.find_one({"_id": _id})

    if existing_doc:
        # Step 4: Build the update query dynamically
        update_query = {"$push": {}}

        for key, values in data.items():
            if isinstance(values, list):  # Ensure we only append lists
                update_query["$push"][key] = {"$each": values}

        # If no valid keys exist, do nothing
        if not update_query["$push"]:
            print("No valid fields to update.")
            return

        # Step 5: Perform the update
        update_result = collection.update_one({"_id": _id}, update_query)

        # Step 6: Provide feedback
        if update_result.matched_count > 0:
            print(f"Successfully appended data to document with _id: {_id}")
        else:
            print(f"Document with _id: {_id} not found.")
    else:
        print(f"Document with _id: {_id} not found. Inserting new document...")
        data["_id"] = _id  # Ensure _id is added to data before inserting
        push_data(database, sub_collection, data)  # Call push_data to insert new document


if __name__ == '__main__':
    data = {
        "name": "John Doe",
        "age": [30, 25, 35],
        "city": "New York"
    }
    
    pull_data(database = "AlphaBet", sub_collection = "BTC", _id = "SVI_IV")
    # push_data(database = "AlphaBet", sub_collection = "Test", data = data)
