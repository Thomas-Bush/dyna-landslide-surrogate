// Set up variables to track the number of entities
var nstate = GetNumberOf(STATE);  
var nnode  = GetNumberOf(NODE);


// Save files every nth timestep   
var timestepSkip = 1;

// Function to setup file for solid velocity output
function setupNodeVelFile(){

  // Create file  
  var localFile = new File("./DATA/NodeVel.csv", File.WRITE);
  
  // Write headers
  localFile.Write("$t,");   

  // Loop through nodes and write ID headers
  for (var nn = 1; nn < nnode - 1; nn++){
    localFile.Write("M1 Node " + internalID + ",");
  }

  // Write final node ID header 
  var internalID = GetLabel(NODE, nnode - 1);
  localFile.Write("M1 Node " + internalID + "\n");

  return localFile;  
}

// Function to write node velocities to file at each timestep
function saveNodeResVels(){

  // Loop through timesteps
  for (var kk = 1; kk < nstate; kk += timestepSkip){

    // Get timestep info
    SetCurrentState(kk);
    time = GetTime(kk);

    // Write timestamp
    posFiles[3].Write(time + ",");     

    // Loop through nodes    
    for (var nn = 1; nn < nnode -1; nn++){
     
      // Get node velocity   
      var ResultantVel = GetData(VM, NODE, nn);
                                
      // Write to file       
      posFiles[3].Write(ResultantVel + ",");     
    }

    // Write final node velocity
    var ResultantVel = GetData(VM, NODE, nnode -1);
    posFiles[3].Write(ResultantVel + "\n");

  }

}