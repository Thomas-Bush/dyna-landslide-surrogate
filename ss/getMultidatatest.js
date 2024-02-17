function getCenterPointsAndAverageDataOfPartElements(model, partID, numberOfStates, componentType) {
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
  
      // Array to collect node indices for data fetch
      let nodeIndices = topology.map(node => node.index);
  
      // Iterate through each node in the element's topology to get the coordinates
      nodeIndices.forEach(nodeIndex => {
        const nodeObject = Node.GetFromIndex(model, nodeIndex);
        if (nodeObject) {
          // Get X, Y, Z coordinates
          const coords = nodeObject.Coordinates();
          sumX += coords[0];
          sumY += coords[1];
          sumZ += coords[2];
        }
      });
  
      // Calculate the averages for coordinates
      const avgX = sumX / topology.length;
      const avgY = sumY / topology.length;
      const avgZ = sumZ / topology.length;
  
      // Now iterate through each state to get the data
      for (let stateIndex = 0; stateIndex < numberOfStates; stateIndex++) {
        // Set the model state to the current state
        model.state = stateIndex + 1;
  
        // Call GetMultipleData for the current state
        const stateData = Node.GetMultipleData(componentType, nodeIndices);
  
        // Iterate through the node indices and accumulate the data for the current state
        nodeIndices.forEach(nodeIndex => {
          const nodeDataValue = stateData[nodeIndex] !== undefined ? stateData[nodeIndex] : 0;
          stateDataSums[stateIndex] += nodeDataValue;
        });
      }
  
      // Prepare the element data with coordinates
      const elementData = { Index: elementIndex, X: avgX, Y: avgY, Z: avgZ };
  
      // Calculate average data for each state and add to the element data
      stateDataSums.forEach((sum, index) => {
        elementData[`State ${index + 1}`] = sum / topology.length;
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