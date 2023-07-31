angular.module('indexApp', [])
  .controller('indexCtlr', function($scope) {
    $scope.yourName = "Tam";
    

    client = new Paho.MQTT.Client("broker.hivemq.com", Number(8000), "ECE568_project");

    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = onMessageArrived;

    client.connect({onSuccess:onConnect});
    function onConnect() {
      console.log("onConnect");
      client.subscribe("/ece568/sensor/rgb");
    }

    function onConnectionLost(responseObject) {
      if (responseObject.errorCode !== 0) {
        console.log("onConnectionLost:"+responseObject.errorMessage);
      }
    }

    function onMessageArrived(message) {
      
      var raw_data = JSON.parse(message.payloadString);
      console.log("onMessageArrived:"+raw_data);
      $scope.$apply(function(){
        $scope.red = raw_data.red;
        $scope.green = raw_data.green;
        $scope.blue = raw_data.blue;
        $scope.clear = raw_data.clear;
      })
    }
  });