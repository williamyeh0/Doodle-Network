var canvas = document.getElementById("paint");
var ctx = canvas.getContext("2d");
var width = canvas.width;
var height = canvas.height;
var curX, curY, prevX, prevY;
var hold = false;
ctx.lineWidth = 15;
var fill_value = true;
var stroke_value = false;
var canvas_data = {"pencil": [], "eraser": []}
                        
               
function reset(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas_data = { "pencil": [], "eraser": [] }
    console.log(canvas_data)
}
        
// pencil tool
        
function pencil(){
        
    canvas.onmousedown = function(e){
        curX = e.clientX - canvas.offsetLeft;
        curY = e.clientY - canvas.offsetTop;
        hold = true;
            
        prevX = curX;
        prevY = curY;
        ctx.lineWidth = 15; // draw width
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
    };
        
    canvas.onmousemove = function(e){
        if(hold){
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw();
        }
    };
        
    canvas.onmouseup = function(e){
        hold = false;
        console.log(canvas_data)
    };
        
    canvas.onmouseout = function(e){
        hold = false;
    };
        
    function draw(){
        ctx.strokeStyle = "#f0f0f0";
        ctx.lineTo(curX, curY);
        ctx.stroke();
        canvas_data.pencil.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
        getPredictionData();
    }
}

// eraser tool
        
function eraser(){
    
    canvas.onmousedown = function(e){
        curX = e.clientX - canvas.offsetLeft;
        curY = e.clientY - canvas.offsetTop;
        hold = true;
            
        prevX = curX;
        prevY = curY;
        ctx.lineWidth = 50; // erase width
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
    };
        
    canvas.onmousemove = function(e){
        if(hold){
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw();
        }
    };
        
    canvas.onmouseup = function(e){
        hold = false;
        console.log(canvas_data)
    };
        
    canvas.onmouseout = function(e){
        hold = false;
    };
        
    function draw(){
        ctx.lineTo(curX, curY);
        ctx.strokeStyle = "#000000";
        ctx.stroke();
        canvas_data.eraser.push({ "startx": prevX, "starty": prevY, "endx": curX, "endy": curY, "thick": ctx.lineWidth, "color": ctx.strokeStyle });
        getPredictionData();
    }    
}  

function getPredictionData(){
    // const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL();

    fetch('/paint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
    })
    .then(response => response.json())
    .then(data => {
        // console.log('Prediction:', typeof(data), data);
        displayPredictionResults(data);
    })
    .catch(error => console.error('Error:', error));
}

function displayPredictionResults(predictionObject) {
    // predictionObject is already sorted dict {class : probability}

    // grab html
    const resultsContainer = document.getElementById('predictionResults');
    resultsContainer.innerHTML = ''; // Clear previous results

    // console.log("predictions in display(): ", typeof(predictionObject), predictionObject) //string
    
    // Create a table to display results
    const table = document.createElement('table');
    table.innerHTML = `
        <tr>
            <th>Class</th>
            <th>Probability</th>
        </tr>
    `;

    predictionObject.forEach(([className, probability]) => {
        const row = table.insertRow();
        row.innerHTML = `
            <td>${className}</td>
            <td>${(probability * 100).toFixed(4)}%</td>
        `;
    });
    resultsContainer.appendChild(table);
}

function save(){
    var filename = document.getElementById("fname").value;
    var data = JSON.stringify(canvas_data);
    var image = canvas.toDataURL();
    
    $.post("/", { save_fname: filename, save_cdata: data, save_image: image });
    alert(filename + " saved");
} 