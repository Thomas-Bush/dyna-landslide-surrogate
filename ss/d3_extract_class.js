class LSDynaDataExtractor {
    constructor() {
        this.model = Model.GetFromID(1);
        this.nState = GetNumberOf(STATE);
        this.topoPartID = 1;
        this.lidPartID = 2;
        this.gatePartID = 3;
        this.airPartID = 4;
        this.debrisPartID = 5;
        this.allNodes = Node.GetAll(this.model);
        this.allSolids = Solid.GetAll(this.model); // Populate allSolids similarly to allNodes
    }

    extractNodes() {
        let csvContent = "Index,Label,X,Y,Z\n";
        this.allNodes.forEach(node => {
            var coords = node.Coordinates();
            if (coords !== null) {
                csvContent += `${node.index},${node.label},${coords[0]},${coords[1]},${coords[2]}\n`;
            }
        });
        return csvContent;
    }

    extractSolids() {
        let csvContent = "Index,Label,Part,n1,n2,n3,n4,n5,n6,n7,n8\n"; // CSV header
        this.allSolids.forEach(solid => {
            var topology = solid.Topology();
            var partLabel = solid.part.label; // Assuming you want the part label
            // Map the topology node objects to their labels
            var nodeLabels = topology.map(node => node.label);
            // Ensure that we have 8 node labels, pad with empty strings if necessary
            var paddedNodeLabels = nodeLabels.concat(new Array(8 - nodeLabels.length).fill(''));
            csvContent += `${solid.index},${solid.label},${partLabel},${paddedNodeLabels.join(',')}\n`;
        });
        return csvContent;
    }

    extractShells() {
        let csvContent = "Index,Label,Part,n1,n2,n3,n4\n"; // CSV header for shells
        const allShells = Shell.GetAll(this.model); // Assuming you can get all shells similarly to how you get all nodes and solids

        allShells.forEach(shell => {
            var topology = shell.Topology();
            var partLabel = shell.part.label; // Assuming you want the part label
            // Map the topology node objects to their labels
            var nodeLabels = topology.map(node => node.label);
            // Ensure that we have 4 node labels, pad with empty strings if necessary
            var paddedNodeLabels = nodeLabels.concat(new Array(4 - nodeLabels.length).fill(''));
            csvContent += `${shell.index},${shell.label},${partLabel},${paddedNodeLabels.join(',')}\n`;
        });
        return csvContent;
    }

    extractStates() {
        let csvContent = "Index,Timestamp\n"; // CSV header for states
        const numStates = this.model.states; // Retrieve the total number of states

        for (let stateIndex = 0; stateIndex < numStates; stateIndex++) {
            const timestamp = this.model.Time(stateIndex); // Get the time for the current state
            csvContent += `${stateIndex},${timestamp}\n`; // Append the state index and timestamp to the CSV content
        }

        return csvContent;
    }

    extractNodalVelocities() {
        const uniqueNodes = new Map();
        const numStates = this.model.states;
        const c = Component.VM;

        // Gather all unique node labels from the solids' topology
        this.allSolids.forEach(solid => {
            const topologyNodes = solid.Topology();
            topologyNodes.forEach(node => {
                if (!uniqueNodes.has(node.label)) {
                    uniqueNodes.set(node.label, node);
                }
            });
        });

        // Convert uniqueNodes to array of node objects
        const solidNodes = Array.from(uniqueNodes.values());

        // Initialize CSV string with headers
        let csvContent = "Node Label";
        for (let i = 1; i <= numStates; i++) {
            csvContent += `,${i}`;
        }
        csvContent += "\n";

        // Initialize an object to store velocity data for each node
        let nodesVelocityData = {};
        solidNodes.forEach(node => {
            nodesVelocityData[node.label] = [node.label]; // Start with the label
        });

        // Loop through each state, update all solidNodes, then loop through each node to collect the data
        for (let stateIndex = 0; stateIndex < numStates; stateIndex++) {
            this.model.state = stateIndex; // Set the model to the current state
            Node.GetMultipleData(c, solidNodes); // Updates the 'data' property of all nodes
            
            solidNodes.forEach(node => {
                // Append the node's velocity magnitude to its array in the object
                if (node.data !== null) {
                    nodesVelocityData[node.label].push(node.data);
                } else {
                    // If there's no data, append a placeholder
                    nodesVelocityData[node.label].push('');
                }
            });
        }

       
        // Convert nodesVelocityData to CSV format
        for (let [label, velocities] of Object.entries(nodesVelocityData)) {
            csvContent += velocities.join(',') + "\n";
        }

        return csvContent;
    }



    extractSolidThicknesses() {
        const numStates = this.model.states;
        const solids = this.allSolids; // Assuming this.allSolids is an array of solid objects

        // Initialize CSV string with headers
        let csvContent = "Solid Label";
        for (let i = 1; i <= numStates; i++) {
            csvContent += `,State ${i}`;
        }
        csvContent += "\n";

        // Initialize an object to store thickness data for each solid
        let solidsThicknessData = {};
        solids.forEach(solid => {
            solidsThicknessData[solid.label] = [solid.label]; // Start with the label
        });

        // Loop through each state, get the thickness for all solids, then collect the data
        for (let stateIndex = 0; stateIndex < numStates; stateIndex++) {
            this.model.state = stateIndex; // Set the model to the current state
            Solid.GetMultipleData(Component.SOX, solids, { extra: 3 }); // Update the 'data' property of all solids
            
            solids.forEach(solid => {
                // Append the solid's thickness data to its array in the object
                if (solid.data !== null) {
                    solidsThicknessData[solid.label].push(solid.data);
                } else {
                    // If there's no data, append a placeholder
                    solidsThicknessData[solid.label].push('');
                }
            });
        }

        // Convert solidsThicknessData to CSV format
        for (let [label, thicknesses] of Object.entries(solidsThicknessData)) {
            csvContent += thicknesses.join(',') + "\n";
        }

        return csvContent;
    }


    writeCSVToFile(csvContent, fileName) {
        var file = new File(`RAW_DATA/${fileName}`, File.WRITE);
        file.Writeln(csvContent);
        file.Close();
    }

    runAllExtractions() {
        if (!File.IsDirectory("RAW_DATA")) {
            File.Mkdir("RAW_DATA");
        }
        this.writeCSVToFile(this.extractNodes(), "nodes.csv");
        this.writeCSVToFile(this.extractSolids(), "solids.csv");
        this.writeCSVToFile(this.extractShells(), "shells.csv");
        this.writeCSVToFile(this.extractStates(), "states.csv");
        this.writeCSVToFile(this.extractNodalVelocities(), "nodal_velocities.csv");
        this.writeCSVToFile(this.extractSolidThicknesses(), "solid_thicknesses.csv");
        // Add other extraction methods as needed
    }
}

// Usage:
const extractor = new LSDynaDataExtractor();
extractor.runAllExtractions();