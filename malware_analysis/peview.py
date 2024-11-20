from os import listdir

import pefile
from os.path import isfile, join

dirs = ["BenignSamples", "MaliciousSamples"]

def preprocessImports(listOfDLLs):
    preprocessedListOfDLLs = []
    return [x.decode().split(".")[0].lower() for x in listOfDLLs]

def getImports(pe):
    listOfImports = []
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        listOfImports.append(entry.dll)
    return preprocessImports(listOfImports)

def getSectionNames(pe):
    listOfSectionNames = []
    for eachSection in pe.sections:
        refined_name = eachSection.Name.decode().replace('\x00','').lower()
        listOfSectionNames.append(refined_name)
    return listOfSectionNames

importsCorpus = []
numSections = []
sectionNames = []

for datasetPath in dirs:
    samples = [f for f in listdir(datasetPath) if isfile(join(datasetPath,f))]
    for file in samples:
        filePath = datasetPath+"/"+file
        try:
            pe = pefile.PE(filePath)
            imports = getImports(pe)
            nSections = len(pe.sections)
            secNames = getSectionNames(pe)
            importsCorpus.append(imports)
            numSections.append(nSections)
            sectionNames.append(secNames)
        except:
            pass



#myPeFile = pefile.PE("C:\\Users\\jafar\\Downloads\\PEview\\PEview.exe")
#print(myPeFile.OPTIONAL_HEADER.AddressOfEntryPoint)
#print(myPeFile.dump_info())

