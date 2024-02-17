SetCurrentModel(1);
var nstate = GetNumberOf(STATE);
var nbeam  = GetNumberOf(BEAM);
var npart  = GetNumberOf(PART);
var nnode  = GetNumberOf(NODE);
var nsolid = GetNumberOf(SOLID);

var objElem;
var time;

// Save every nth timestep.
var timestepSkip = 1;

//
var skipSomeFiles = 1;

// Get current working folder, for filenaming purposes
var fullPath = GetCurrentDirectory();
var dir = fullPath.replace(/^.*[\\\/]/, '');

System("mkdir DATA");

// Merge with function below 
function setupSolidFiles(filename)
{
	var localFile  = new File("./DATA\\" + filename, File.WRITE);
	localFile.Write("$t,");

	for (var bb = 1; bb < nsolid - 1; bb++)
	{
		var internalID = GetLabel(SOLID, bb);
		localFile.Write("M1 Solid " + internalID + ",");
	}

	var internalID = GetLabel(SOLID, nsolid - 1);
	localFile.Write("M1 Solid " + internalID + "\n");
	
	return localFile;
}

function setupNodeVelFile()
{
	var localFile  = new File("./DATA\\" + "NodeVel.csv", File.WRITE);
	localFile.Write("$t,");

	for (var nn = 1; nn < nnode - 1; nn++)
	{
		var internalID = GetLabel(NODE, nn);
		localFile.Write("M1 Node " + internalID + ",");
	}

	var internalID = GetLabel(NODE, nnode - 1);
	localFile.Write("M1 Node " + internalID + "\n");
	
	return localFile;
}

var posFiles = new Array();

if (skipSomeFiles == 0)
{
	posFiles[0] = new File("./DATA\\" + "maxPositions.csv", File.WRITE);
	posFiles[0].Write("$Time,xMax,yMax,zMax\n");
}
posFiles[1] = new File("./DATA\\" + "systemNodes.csv", File.WRITE);
for (var i = 0; i < 7; i++) posFiles[1].Write("$\n");
posFiles[1].Write("$ Entity ID,X Basic,Y Basic,Z Basic\n");

posFiles[2] = setupSolidFiles("Solid_Extra_3.csv");

posFiles[3] = setupNodeVelFile();

posFiles[4]  = new File("./DATA\\" + "SolidCoords.csv", File.WRITE);
for (var i = 0; i < 7; i++) posFiles[4].Write("$\n");
posFiles[4].Write("$Entity ID,x,y,z\n");

if (skipSomeFiles == 0)
{
	posFiles[5]  = new File("./DATA\\" + "systemNodesIdx.csv", File.WRITE);
	posFiles[5].Write("$xy\n");

	posFiles[6]  = new File("./DATA\\" + "SolidCoordsIdx.csv", File.WRITE);
	posFiles[6].Write("$xy\n");

	posFiles[7] = setupSolidFiles("SolidVel.csv");
}

function saveSystemNodes()
{
	kk = 1;
	SetCurrentState(kk);
	time = GetTime(kk);

	for (var nn = 1; nn < nnode; nn++) // nnode
	{
		var nodeID = GetLabel(NODE, nn);

		var nodePropertiesX = GetData(BX, NODE, nn);	
		var nodePropertiesY = GetData(BY, NODE, nn);	
		var nodePropertiesZ = GetData(BZ, NODE, nn);							
		posFiles[1].Write('N' + nodeID + "," +
				Math.round(nodePropertiesX) + "," + 
				Math.round(nodePropertiesY) + "," + 
				nodePropertiesZ + "\n");	
		if (skipSomeFiles == 0)
		{
			posFiles[5].Write(
					Math.round(nodePropertiesX).toString() + 
					Math.round(nodePropertiesY).toString() +  "\n");	
		}
	}
	for (var i = 0; i < 2; i++) posFiles[1].Write("$\n");
	posFiles[1].Write("\n")
}

function saveFlowDepth()
{
	for (var kk = 1; kk < nstate; kk += timestepSkip)
	{
		SetCurrentState(kk);
		time = GetTime(kk);
		
		posFiles[2].Write(time + ",");	
		for (var bb = 1; bb < nsolid -1; bb++)
		{
			var ALE_properties = GetData(SOX,  SOLID, bb, 0, 3);
							
			posFiles[2].Write(
					ALE_properties + "," );	
		}
		var ALE_properties = GetData(SOX,  SOLID, nsolid -1, 0, 3);
					posFiles[2].Write(
					ALE_properties + "\n" );					

	}
}

function saveNodeResVels()
{
	for (var kk = 1;  kk < nstate; kk += timestepSkip)
	{
		SetCurrentState(kk);
		time = GetTime(kk);
		posFiles[3].Write(time + ",");	

		for (var nn = 1; nn < nnode -1; nn++)
		{
			var ResultantVel = GetData(VM, NODE, nn);
							
			posFiles[3].Write(
					ResultantVel + "," );	
		}
		var ResultantVel = GetData(VM, NODE, nnode -1);
		posFiles[3].Write(ResultantVel + "\n" );					

	}
}

function saveSolidCoords()
{
	kk = 1;
	SetCurrentState(kk);
	time = GetTime(kk);

	for (var bb = 1; bb < nsolid; bb++)
	{
		var internalID = GetLabel(SOLID, bb);
		var a = GetTopology(SOLID, bb);
		var nnodes_loc = a.nn;

		const nodeys = [];

		nodeys[0] = a.top[0];
		nodeys[1] = a.top[1];
		nodeys[2] = a.top[2];
		nodeys[3] = a.top[3];
		nodeys[4] = a.top[4];
		nodeys[5] = a.top[5];
		nodeys[6] = a.top[6];
		nodeys[7] = a.top[7];
		
		var localNodeX = 0;
		var localNodeY = 0;
		var localNodeZ = 0;

		for (var aa = 0; aa < 8; aa++)
		{
			localNodeX += GetData(BX, NODE, nodeys[aa]);	
			localNodeY += GetData(BY, NODE, nodeys[aa]);	
			localNodeZ += GetData(BZ, NODE, nodeys[aa]);	
		}
		
		localNodeX /= 8;
		localNodeY /= 8;
		localNodeZ /= 8;	
		
		posFiles[4].Write("H" +
				internalID + "," +
				Math.round(localNodeX) + "," + 
				Math.round(localNodeY) + "," + 
				localNodeZ + "\n");	
		if (skipSomeFiles == 0)
		{
			posFiles[6].Write(
				Math.round(localNodeX).toString() + 
				Math.round(localNodeY).toString() +  "\n");	
		}
	}
	for (var i = 0; i < 2; i++) posFiles[4].Write("$\n");
	posFiles[4].Write("\n")
}

function saveSolidVels()
{
	const posProperties = new Array();
	for (var kk = 1;  kk < nstate; kk += timestepSkip)
	{
		SetCurrentState(kk);
		time = GetTime(kk);

		posProperties[0] = 1000000;
		posProperties[1] = 0;
		posProperties[2] = 1000000;

		if (skipSomeFiles == 0)
		{
			posFiles[7].Write(time + ",");	
		}
		
		for (var bb = 1; bb < nsolid; bb++)
		{
			var internalID = GetLabel(SOLID, bb);
			var a = GetTopology(SOLID, bb);
			var nnodes_loc = a.nn;

			const nodeys = [];

			nodeys[0] = a.top[0];
			nodeys[1] = a.top[1];
			nodeys[2] = a.top[2];
			nodeys[3] = a.top[3];
			nodeys[4] = a.top[4];
			nodeys[5] = a.top[5];
			nodeys[6] = a.top[6];
			nodeys[7] = a.top[7];
			
			var localNodeX = 0;
			var localNodeY = 0;
			var localNodeZ = 0;
			var localVelR  = 0;

			for (var aa = 0; aa < 8; aa++)
			{
				localVelR  += GetData(VM, NODE, nodeys[aa]);	
			}

			localNodeX += GetData(BX, NODE, nodeys[0]);	
			localNodeY += GetData(BY, NODE, nodeys[0]);	
			localNodeZ += GetData(BZ, NODE, nodeys[0]);	
			
			localVelR  /= 8;	
			
			if (skipSomeFiles == 0)
			{
				posFiles[7].Write(localVelR);	
				
				if (bb < nsolid - 1)
				{
					posFiles[7].Write(",");	
				}
				else
				{
					posFiles[7].Write("\n");	
				}
			}
			
			if (localVelR > 0)
			{
				if (localNodeX < posProperties[0])
				{
					posProperties[0] = localNodeX;
				}
				if (localNodeY > posProperties[1])
				{
					posProperties[1] = localNodeY;
				}
				if (localNodeZ < posProperties[2])
				{
					posProperties[2] = localNodeZ;
				}
			}
		}
		
		if (skipSomeFiles == 0)
		{
			if (kk > 1)
			{
				posFiles[0].Write(time + "," +
				posProperties[0] + "," + 
				posProperties[1] + "," +
				posProperties[2] + "\n");	
			}
		}
	}
}

saveSystemNodes();
saveFlowDepth();
saveNodeResVels();
saveSolidCoords();
saveSolidVels();
Print("script done!");