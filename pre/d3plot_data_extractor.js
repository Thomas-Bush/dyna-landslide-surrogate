// memory: 500

/**
 * Class to extract data from an LS-DYNA model and output to CSV files.
 */
class D3PlotDataExtractor {

    /**
     * Constructor. 
     * Gets the active model, number of states, all nodes, solids, and shells.
     */
    constructor() {
  
      // Get the active model
      this.model = Model.GetFromID(1);
  
      // Get the number of states 
      this.numStates = this.model.states;
  
      // Get all nodes
      this.allNodes = Node.GetAll(this.model);
  
      // Get all solids
      this.allSolids = Solid.GetAll(this.model);
  
      // Get all shells
      this.allShells = Shell.GetAll(this.model);
  
    }
  
    /**
     * Extracts node data to a CSV string.
     * @returns {string} CSV string containing node index, label, and coordinates.
     */  
    extractNodes() {
      
      // CSV header
      let csvContent = "Node_Label,Node_X,Node_Y,Node_Z\n";
  
      // Loop through nodes
      this.allNodes.forEach(node => {
  
        // Get node coordinates  
        var coords = node.Coordinates();
  
        // Check if coordinates exist
        if (coords !== null) {
          
          // Append node data to CSV content
          csvContent += `${node.label},${coords[0]},${coords[1]},${coords[2]}\n`;
  
        }
  
      });
  
      return csvContent;
  
    }
  
    /** 
     * Extracts solid element data to a CSV string.
     * @returns {string} CSV string containing solid index, label, part, and node labels.
     */
    extractSolids() {
      
      // CSV header 
      let csvContent = "Solid_Label,Solid_Part,Solid_N1,Solid_N2,Solid_N3,Solid_N4,Solid_N5,Solid_N6,Solid_N7,Solid_N8\n";
  
      // Loop through solids
      this.allSolids.forEach(solid => {
  
        // Get topology
        var topology = solid.Topology();
        
        // Get part label
        var partLabel = solid.part.label;
  
        // Get node labels from topology  
        var nodeLabels = topology.map(node => node.label);
  
        // Pad node labels to 8
        var paddedNodeLabels = nodeLabels.concat(new Array(8 - nodeLabels.length).fill(''));
  
        // Append to CSV content
        csvContent += `${solid.label},${partLabel},${paddedNodeLabels.join(',')}\n`;
  
      });
  
      return csvContent;
  
    }
  
    /**
     * Extracts shell element data to a CSV string. 
     * @returns {string} CSV string containing shell index, label, part, and node labels.
     */
    extractShells() {
      
      // CSV header
      let csvContent = "Shell_Label,Shell_Part,Shell_N1,Shell_N2,Shell_N3,Shell_N4\n";
  
      // Loop through shells
      this.allShells.forEach(shell => {
  
        // Get topology
        var topology = shell.Topology();
        
        // Get part label
        var partLabel = shell.part.label;
  
        // Get node labels from topology
        var nodeLabels = topology.map(node => node.label);
  
        // Pad node labels to 4
        var paddedNodeLabels = nodeLabels.concat(new Array(4 - nodeLabels.length).fill(''));
  
        // Append to CSV content
        csvContent += `${shell.label},${partLabel},${paddedNodeLabels.join(',')}\n`;
  
      });
  
      return csvContent;
  
    }
  
    /**
     * Extracts state data to a CSV string.
     * @returns {string} CSV string containing state index and timestamp.
     */
    extractStates() {
      
      // CSV header
      let csvContent = "State_Label,State_Timestamp\n";
  
      // Loop through states
      for (let stateIndex = 0; stateIndex < this.numStates; stateIndex++) {
  
        // Get timestamp
        const timestamp = this.model.Time(stateIndex);
  
        // Append state data  
        csvContent += `${stateIndex + 1},${timestamp}\n`;
  
      }
  
      return csvContent;
  
    }
  
    /**
     * Extracts nodal velocity data to a CSV string.
     * @returns {string} CSV string containing node labels and velocity magnitude for each state.
     */
    extractNodalVelocities() {
  
      // Map to store unique nodes
      const uniqueNodes = new Map();
  
      // Number of states
  
      const numStates = this.model.states;
  
      // Velocity component
      const c = Component.VM;
  
      // Get unique nodes from solid topology 
      this.allSolids.forEach(solid => {
        const topologyNodes = solid.Topology();
        topologyNodes.forEach(node => {
          if (!uniqueNodes.has(node.label)) {
            uniqueNodes.set(node.label, node);
          }
        });
      });
  
      // Convert to array
      const solidNodes = Array.from(uniqueNodes.values());
  
      // CSV header
      let csvContent = "Node_Label";
      for (let i = 1; i <= numStates; i++) {
        csvContent += `,State_${i}`;
      }
      csvContent += "\n";
  
      // Object to store velocity data
      let nodesVelocityData = {};
      solidNodes.forEach(node => {
        nodesVelocityData[node.label] = [node.label];
      });
  
      // Loop through states
      for (let stateIndex = 0; stateIndex < numStates; stateIndex++) {
  
        // Set state  
        this.model.state = stateIndex;
  
        // Update node data
        Node.GetMultipleData(c, solidNodes);
  
        // Collect data
        solidNodes.forEach(node => {
          if (node.data !== null) {
            nodesVelocityData[node.label].push(node.data);
          } else {
            nodesVelocityData[node.label].push('');
          }
        });
  
      }
  
      // Build CSV content
      for (let [label, velocities] of Object.entries(nodesVelocityData)) {
        csvContent += velocities.join(',') + "\n";
      }
  
      return csvContent;
  
    }
  
    /**
     * Extracts solid thickness data to a CSV string.
     * @returns {string} CSV string containing solid labels and thickness for each state.
     */
    extractSolidThicknesses() {
  
      // Number of states
      const numStates = this.model.states;
  
      // All solids
      const solids = this.allSolids; 
  
      // CSV header
      let csvContent = "Solid_Label";
      for (let i = 1; i <= numStates; i++) {
        csvContent += `,State_${i}`;
      }
      csvContent += "\n";
  
      // Object to store thickness data
      let solidsThicknessData = {};
      solids.forEach(solid => {
        solidsThicknessData[solid.label] = [solid.label]; // Label
      });
  
      // Loop through states
      for (let stateIndex = 0; stateIndex < numStates; stateIndex++) {
  
        // Set state
        this.model.state = stateIndex;
  
        // Update solid data
        Solid.GetMultipleData(Component.SOX, solids, { extra: 3 });
  
        // Collect data
        solids.forEach(solid => {
          if (solid.data !== null) {
            solidsThicknessData[solid.label].push(solid.data);
          } else {
            solidsThicknessData[solid.label].push(''); 
          }
        });
  
      }
  
      // Build CSV content
      for (let [label, thicknesses] of Object.entries(solidsThicknessData)) {
        csvContent += thicknesses.join(',') + "\n";
      }
  
      return csvContent;
  
    }
  
    /**
     * Writes a CSV string to a file.
     * @param {string} csvContent - The CSV data
     * @param {string} fileName - The file name
     */
    writeCSVToFile(csvContent, fileName) {
      // Get the current working directory
      var cwd = GetCurrentDirectory();
      
      // Extract the folder name from the current working directory
      // Assuming the directory path is separated by '/' or '\\'
      var folderName = cwd.split(/[/\\]/).pop();
      
      // Append folder name to file name
      fileName = folderName + "_" + fileName;

      // Create file in the RAW_DATA directory with the modified file name
      var file = new File(`RAW_DATA/${fileName}`, File.WRITE);

      // Write content 
      file.Writeln(csvContent);

      // Close file
      file.Close();
    }
  
    /**
     * Runs all data extraction methods and writes CSV files.
     */
    runAllExtractions() {
  
      // Create output directory if needed  
      if (!File.IsDirectory("RAW_DATA")) {
        File.Mkdir("RAW_DATA");
      }
  
      // Extract and write all CSVs
      this.writeCSVToFile(this.extractNodes(), "nodes.csv");
      this.writeCSVToFile(this.extractSolids(), "solids.csv");
      this.writeCSVToFile(this.extractShells(), "shells.csv");
      this.writeCSVToFile(this.extractStates(), "states.csv");
      this.writeCSVToFile(this.extractNodalVelocities(), "nodal_velocities.csv");
      this.writeCSVToFile(this.extractSolidThicknesses(), "solid_thicknesses.csv");
  
    }
  
  }

  // Usage:
const extractor = new D3PlotDataExtractor();
extractor.runAllExtractions();