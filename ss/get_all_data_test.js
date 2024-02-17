function getCenterPointsAndAverageDataOfPartElements(model, partID, numberOfStates, component) {
    // Define the headers for the coordinates
    let headers = ["Index", "X", "Y", "Z"];
  
    // Add headers for each state
    for (let i = 1; i <= numberOfStates; i++) {
      headers.push(`State ${i}`);
    }
  
    // Get part from model using the part ID
    const part = Part.GetFromID(model, partID);
  
    // Get elements that make up the part
    const partElements = part.Elements();
  
    // Array to hold the data for each element
    const elementsData = partElements.map((element, elementIndex) => {
      const topology = element.Topology(); // Get topology of the element
  
      // Initialize sums for x, y, z coordinates
      let sumX = 0, sumY = 0, sumZ = 0;
  
      // Initialize arrays to accumulate data for each state
      let stateDataSums = new Array(numberOfStates).fill(0);
  
      // Array to collect nodes for data fetch
      let nodes = topology.map(node => Node.GetFromIndex(model, node.index));
  
      // Filter out any undefined node references
      nodes = nodes.filter(node => node !== undefined);
  
      // Iterate through each node to get the coordinates
      nodes.forEach(node => {
        // Get X, Y, Z coordinates
        const coords = node.Coordinates();
        sumX += coords[0];
        sumY += coords[1];
        sumZ += coords[2];
      });
  
      // Calculate the averages for coordinates
      const avgX = sumX / nodes.length;
      const avgY = sumY / nodes.length;
      const avgZ = sumZ / nodes.length;
  
      // Now iterate through each state to get the data
      for (let stateIndex = 0; stateIndex < numberOfStates; stateIndex++) {
        // Set the model state to the current state
        model.state = stateIndex + 1;
  
        // Call GetMultipleData for the current state
        const stateData = Node.GetMultipleData(component, nodes);
  
        // Iterate through the nodes and accumulate the data for the current state
        nodes.forEach(node => {
          const nodeDataValue = stateData[node.index] !== undefined ? stateData[node.index] : 0;
          stateDataSums[stateIndex] += nodeDataValue;
        });
      }
  
      // Prepare the element data with coordinates
      const elementData = { Index: elementIndex, X: avgX, Y: avgY, Z: avgZ };
  
      // Calculate average data for each state and add to the element data
      stateDataSums.forEach((sum, index) => {
        elementData[`State ${index + 1}`] = sum / nodes.length;
      });
  
      return elementData;
    });
  
    // Reset the model state to its original state after processing
    model.state = 1;
  
    // Return the headers and the elements' data
    return {
      headers,
      elementsData
    };
  }

const model = Model.GetFromID(1);

const component = Component.CZ

 
 
const partID = 5



const attempt1 = getCenterPointsAndAverageDataOfPartElements(model, partID, 5, component)

const firstElementKeys = Object.keys(attempt1.elementsData[0]);

Message('Keys of the first element:', firstElementKeys);