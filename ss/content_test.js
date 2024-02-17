const model = Model.GetFromID(1);

const nodeTotal = Node.Total(model);
const solidTotal = Solid.Total(model);
const shellTotal = Shell.Total(model);

Message("Nodes: " + nodeTotal)
Message("Solids: " + solidTotal)
Message("Shells: " + shellTotal)





const solid = Solid.GetAll(model)

const solid1 = solid[1]

const sol_top = solid1.Topology()

Message(sol_top)

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


  const debrisNodes = getAllPartNodes(model, 5)

  Message("no of deb nodes" + debrisNodes.length)

  Node.GetMultipleData(Component.VM, debrisNodes)

  Message("data check " + debrisNodes[1].data)