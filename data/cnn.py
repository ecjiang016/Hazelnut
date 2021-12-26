import pickle

def load(name):
    infile = open("Training Data CNN", 'rb')
    SavedData = pickle.load(infile)
    infile.close()
    output = {}
    try:
        index = SavedData["Name"].index(name)
        for data in SavedData:
            output[data] = SavedData[data][index]
        return output
    except:
        print("Loading failed")
        return None

def delete(name):
    #Read file to find delete item index
    infile = open("Training Data CNN", 'rb')
    SavedData = pickle.load(infile)
    infile.close()

    #Deleting the data
    output = {} #outputs the deleted item
    try:
        index = SavedData["Name"].index(name)
        for data in SavedData:
            output[data] = SavedData[data].pop(index)

        #Writing new dict into the file
        outfile = open("Training Data CNN", 'wb')
        pickle.dump(SavedData, outfile)
        outfile.close()

        return output
            
    except:
        print("Deletion failed")
        return None

def replace(name, layout):
    #Reformatting Data
    temp_array = {"Name":name, "Layout":layout}

    #Load data and get index
    infile = open("Training Data CNN", 'rb')
    SavedData = pickle.load(infile)
    infile.close()
    try:
        index = SavedData["Name"].index(name)

        #Replacing old data
        for data in SavedData:
            SavedData[data][index] = temp_array[data]

        #Write data
        outfile = open("Training Data CNN", 'wb')
        pickle.dump(SavedData, outfile)
        outfile.close()
        
        
    except:
        print("Replacement failed")
        return None
    
def check(name, data):
    if name in data:
        return check(input("Name taken, try another name: "), data)
    else:
        return name
    
def save(name, layout, auto_replace=False):

    #Check if name is taken 
    infile = open("Training Data CNN", 'rb')
    SavedData = pickle.load(infile)
    if name in SavedData["Name"]:
        if input("Data already exists under this name. Replace? (Y/N) ") == "Y" or auto_replace:
            replace(name, layout)
            return None
        else:
            name = check(name, SavedData["Name"])
    infile.close()
    
    #Reformatting Data
    temp_array = {"Name":name, "Layout":layout}

    #Appending new data
    for data in SavedData:
        SavedData[data].append(temp_array[data])
    
    #Write data
    outfile = open("Training Data CNN", 'wb')
    pickle.dump(SavedData, outfile)
    outfile.close()

def get_all_data():
    infile = open("Training Data CNN", 'rb')
    SavedData = pickle.load(infile)
    infile.close()
    return SavedData

def all_names():
    return get_all_data()["Name"]

if __name__ == "__main__":
    print(get_all_data())