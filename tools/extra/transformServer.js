var http = require('http');
var fs = require('fs');
var transformFileLocation = 'yolo_test_training.png';

http.createServer(function(req, res){
	res.writeHead(200);
	readStream = fs.createReadStream(transformFileLocation);
	readStream.on('data', function(data){
		res.write(data);
	});
	readStream.on('end', function(data){
		res.end();
	});
}).listen(9999);
