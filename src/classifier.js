import React, { useEffect } from 'react';
import * as ml5 from 'ml5';

const options = {
  version: 1,
  topk: 3,
  learningRate: 0.00005,
  hiddenUnits: 100,
  epochs: 100, // random attempt - not sure
  numLabels: 3, // the default was 2
  batchSize: 0.4
};

const featureExtractor = ml5.featureExtractor('MobileNet', options, () =>
  console.log('classifier ready')
);

const classifier = featureExtractor.classification();
console.log(featureExtractor);

const classify = async () => {
  let img = new Image();
  img.src = 'images/paper/0a3UtNzl5Ll3sq8K.png';
  img.width = 224;
  img.height = 224;
  await classifier.addImage(img, 'paper');

  img.src = 'images/paper/0cb6cVL8pkfi4wF6.png';
  await classifier.addImage(img, 'paper');

  img.src = 'images/scissors/0CSaM2vL2cWX6Cay.png';
  await classifier.addImage(img, 'scissors');

  img.src = 'images/scissors/0ePX1wuCc3et7leL.png';
  await classifier.addImage(img, 'scissors');

  img.src = 'images/scissors/1CXgK9fgGdSRggD9.png';
  await classifier.addImage(img, 'scissors');

  img.src = 'images/rock/0bioBZYFCXqJIulm.png';
  await classifier.addImage(img, 'rock');

  img.src = 'images/rock/0NDYNEoDui7o64gU.png';
  await classifier.addImage(img, 'rock');

  img.src = 'images/rock/2NmrcDGkc7FQuu12.png';
  await classifier.addImage(img, 'rock');

  // Retrain the network
  classifier.train(function(lossValue) {
    console.log('Loss is', lossValue);
  });
};

const predict = () => {
  let img = new Image();
  //   img.src = 'images/rock/2NmrcDGkc7FQuu12.png';
  img.src = 'images/scissors/2fxAdPTgrVIoITsL.png';
  img.width = 224;
  img.height = 224;
  // Get a prediction for that image
  classifier.classify(img, function(err, result) {
    console.log(result); // Should output 'dog'
  });
};

const saveModel = () => {
  classifier.save(); // Default name model.json
};

const loadModel = async () => {
  await classifier.load('model/model.json');
  console.log('Model loaded');
};

export default () => {
  useEffect(() => {
    // classify();
  }, []);
  return (
    <div>
      <button onClick={predict}>Predict</button>
      <button onClick={classify}>Train</button>
      <button onClick={saveModel}>Save</button>
      <button onClick={loadModel}>Load</button>
    </div>
  );
};
