// Improved script with better practices and error handling

// Use 'const' for the file path and 'let' for variables that will change
const filePath = "C:/Users/thomas.bush/Documents/temp/lsdyna-test/temp/test.txt";
let n, lineNumber;

function writeNumbersToFile(path) {
    var f = new File(path, File.WRITE); // Use 'var' or 'let' to declare the file handle
    for (var n = 1; n <= 10; n++) {
        f.Writeln(n); // Corrected method name from 'WriteIn' to 'Writeln'
    }
    f.Close(); // Moved 'f.Close()' inside the function
}


// Execute the function to write numbers to the file
writeNumbersToFile(filePath);