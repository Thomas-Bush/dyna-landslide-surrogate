


/* NOTES

ELEMENT TYPE CODES

SOLID = 1
BEAM = 2
SHELL = 3
PART = 33
NODE = -100

*/

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// UTILITY FUNCTIONS
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

/**
 * Retrieves an array of unique Node objects associated with a specific part in a model.
 * It iterates through all elements of a part, collects their node indices,
 * ensures the uniqueness of these indices, and then fetches the corresponding Node objects.
 *
 * @param {Model} model - The model object that contains the part and nodes.
 * @param {number} partID - The identifier for the part within the model.
 * @returns {Node[]} An array of Node objects associated with the part. If a node index does not
 * correspond to an existing Node, it will not be included in the returned array.
 */
function getAllPartNodes(model, partID) {
    // Get part from model using the part ID
    const part = Part.GetFromID(model, partID);
  
    // Get elements that make up the part (assuming shells or solids)
    const partElements = part.Elements();
  
    // Array to hold the unique indices of nodes
    const elementNodeIndices = [];
  
    // Collect unique node indices from the part's shells
    partElements.forEach(partElement => {
      const topology = partElement.Topology(); // Get topology of the shell
  
      // Go through each node in the shell's topology
      topology.forEach(node => {
        const nodeIndex = node.index; // Get node index
  
        // Add the node index if it's not already in the array
        if (!elementNodeIndices.includes(nodeIndex)) {
          elementNodeIndices.push(nodeIndex);
        }
      });
    });
  
    // Array to hold the Node objects
    const elementNodeObjects = [];
  
    // Fetch Node objects from the unique indices
    elementNodeIndices.forEach(index => {
      const node = Node.GetFromIndex(model, index);
      // Add the Node object if it exists
      if (node !== null) {
        elementNodeObjects.push(node);
      }
    });
  
    return elementNodeObjects;
  } 

/**
 * Calculates the center point (average coordinates) of each element within a specified part of the model
 * and returns the result with column headers.
 *
 * @param {Object} model - The model containing parts and elements.
 * @param {number|string} partID - The identifier for the part within the model.
 * @returns {Object} An object containing:
 *                   - `headers`: an array of strings representing the column headers.
 *                   - `elementsData`: an array of objects, each containing the index of the element and the average x, y, z coordinates for the nodes in the element.
 */
function getCenterPointsOfPartElements(model, partID) {
	// Define the headers for the coordinates
	const headers = ["Index", "X", "Y", "Z"];
  
	// Get part from model using the part ID
	const part = Part.GetFromID(model, partID);
  
	// Get elements that make up the part
	const partElements = part.Elements();
  
	// Array to hold the data for each element
	const elementsData = partElements.map((element, elementIndex) => {
	  const topology = element.Topology(); // Get topology of the element
  
	  // Initialize sums for x, y, z coordinates
	  let sumX = 0, sumY = 0, sumZ = 0;
  
	  // Iterate through each node in the element's topology
	  topology.forEach(node => {
		const nodeObject = Node.GetFromIndex(model, node.index);
		if (nodeObject) {
		  // Assuming the node object has a method 'Coordinates' to get X, Y, Z coordinates
		  const coords = nodeObject.Coordinates();
		  sumX += coords[0];
		  sumY += coords[1];
		  sumZ += coords[2];
		}
	  });
  
	  // Calculate the averages
	  const avgX = sumX / topology.length;
	  const avgY = sumY / topology.length;
	  const avgZ = sumZ / topology.length;
  
	  // Return the data for the current element
	  return { Index: elementIndex, X: avgX, Y: avgY, Z: avgZ };
	});
  
	// Return the headers and the elements' data
	return {
	  headers,
	  elementsData
	};
  }






/**
 * Extracts the coordinates of each node object and returns an array of objects with index and coordinates.
 *
 * @param {Object[]} nodeObjects - An array of objects, each with a Coordinates() method
 *                                  that returns an array [x, y, z].
 * @returns {Object[]} An array of objects with the structure { index, x, y, z }.
 */
function getNodeCoords(nodeObjects) {
	return nodeObjects.map((node, index) => {
	  const coords = node.Coordinates();
	  if (coords !== null && Array.isArray(coords) && coords.length === 3) {
		const [x, y, z] = coords;
		return { index, x, y, z };
	  }
	  return null;
	}).filter(item => item !== null); // Filter out any null entries if Coordinates() returned null
  }



  function convertToCSV(headers, data) {
	// Combine the header array into a CSV string
	const csvString = headers.join(',') + '\n';
  
	// Combine the data array into a CSV string
	return data.reduce((acc, row) => {
	  const rowString = headers.map(header => row[header]).join(',');
	  return acc + rowString + '\n';
	}, csvString);
  }



// A function to write a string to a file using the interpreter's File object
function writeToFile(path, filename, content) {
	// Combine the path and filename
	var fullPath = path.endsWith('/') ? path + filename : path + '/' + filename;
  
	// Open the file in write mode
	var file = new File(fullPath, File.WRITE);
  
	// Write the content to the file
	file.Write(content);
  
	// Close the file after writing
	file.Close();
  }

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// MAIN LOGIC
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



// MODEL DEFINITION

const model = Model.GetFromID(1); // set model

const nState = GetNumberOf(STATE); // get number of states


// SET THE PART IDS
/*
  Based on Arup HK modelling practice, the following Part IDs are used:

  1 - Topo 
  2 - Lid 
  3 - Gate
  4 - Air
  5 - Debris

*/

const topoPartID = 1;
const lidPartID = 2;
const gatePartID = 3;
const airPartID = 4;
const debrisPartID = 5;



// GET TOPO XYZ
// X,Y,Z coords are calculated at the center of each topographic shell element

Message("Starting Topo XYZ")

const topoXYZ = getCenterPointsOfPartElements(model, topoPartID)

const topoXYZcsv = convertToCSV(topoXYZ.headers, topoXYZ.elementsData)

const filePath = "C:\\Users\\thomas.bush\\Documents\\temp\\lsdyna-test\\temp"

const topoFileName = "topoXYZ.csv"

writeToFile(filePath, topoFileName, topoXYZcsv)

Message("Finished Topo XYZ")

// GET SOLID XYZ
// X,Y,Z coords are calculated at the center of each solid element

Message("Starting Solid XYZ")

const debrisXYZ = getCenterPointsOfPartElements(model, debrisPartID)

const debrisXYZcsv = convertToCSV(debrisXYZ.headers, debrisXYZ.elementsData)

const debrisFileName = "debrisXYZ.csv"

writeToFile(filePath, debrisFileName, debrisXYZcsv)

Message("Finished solid XYZ")