import React, { useEffect, useState, useRef } from 'react';
import * as ml5 from 'ml5';
import { Upload, Button, Icon } from 'antd';

import model from './myKNN.json';

import paper1 from './images/paper/1yKjzquSvl9ShK7K.png';
import paper2 from './images/paper/0a3UtNzl5Ll3sq8K.png';
import paper3 from './images/paper/0vugygEjxQJPr9yz.png';

import rock1 from './images/rock/0bioBZYFCXqJIulm.png';
import rock2 from './images/rock/0NDYNEoDui7o64gU.png';
import rock3 from './images/rock/2NmrcDGkc7FQuu12.png';
import rock4 from './images/rock/5yHTRIIDcdrXqMYJ.png';

import scissors1 from './images/scissors/0CSaM2vL2cWX6Cay.png';
import scissors2 from './images/scissors/0ePX1wuCc3et7leL.png';
import scissors3 from './images/scissors/1CXgK9fgGdSRggD9.png';

const DropZone = ({ inputImage }) => {
  const [name, setName] = useState(null);

  const setImage = event => {
    console.log(event.target.files);

    setName(URL.createObjectURL(event.target.files[0]));
  };
  return (
    <div>
      <div>
        <input type="file" onChange={setImage} />
        <img
          src={name}
          alt=""
          width="224"
          height="224"
          hidden
          ref={inputImage}
        />
      </div>
    </div>
  );
};

const ModelInput = ({ setUrl }) => {
  const setImage = event => {
    console.log(event.target.files);
    setUrl(URL.createObjectURL(event.target.files[0]));
  };
  return (
    <div>
      <div>
        <input type="file" onChange={setImage} name="as" />
        <label>aassa</label>
      </div>
    </div>
  );
};

export default () => {
  const [knnClassifier, setKnnClassifier] = useState(null);
  const [featureExtractor, setFeatureExtractor] = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const inputRef = useRef();
  const paper1Ref = useRef(null);
  const paper2Ref = useRef(null);
  const paper3Ref = useRef(null);
  const rock1Ref = useRef(null);
  const rock2Ref = useRef(null);
  const rock3Ref = useRef(null);
  const rock4Ref = useRef(null);
  const scissors1Ref = useRef(null);
  const scissors2Ref = useRef(null);
  const scissors3Ref = useRef(null);

  const classify = async () => {
    // TRAIN IMGS
    // Add examples with a label to classifier
    // Paper
    let features = await featureExtractor.infer(paper1Ref.current);
    knnClassifier.addExample(features, 'paper');

    features = await featureExtractor.infer(paper2Ref.current);
    knnClassifier.addExample(features, 'paper');

    // Rock
    features = await featureExtractor.infer(paper3Ref.current);
    knnClassifier.addExample(features, 'paper');

    features = await featureExtractor.infer(rock1Ref.current);
    knnClassifier.addExample(features, 'rock');

    features = await featureExtractor.infer(rock2Ref.current);
    knnClassifier.addExample(features, 'rock');

    features = await featureExtractor.infer(rock3Ref.current);
    knnClassifier.addExample(features, 'rock');

    // Scissors
    features = await featureExtractor.infer(scissors1Ref.current);
    knnClassifier.addExample(features, 'scissors');

    features = await featureExtractor.infer(scissors2Ref.current);
    knnClassifier.addExample(features, 'scissors');

    features = await featureExtractor.infer(scissors3Ref.current);
    knnClassifier.addExample(features, 'scissors');

    console.log(knnClassifier.getNumLabels());
  };

  const predict = async () => {
    const featuresScissors = featureExtractor.infer(scissors3Ref.current);
    let results = await knnClassifier.classify(featuresScissors);
    console.log(results);

    const featuresRock = featureExtractor.infer(rock4Ref.current);
    results = await knnClassifier.classify(featuresRock);
    console.log(results);

    const featuresPaper = featureExtractor.infer(paper1Ref.current);
    results = await knnClassifier.classify(featuresPaper);
    console.log(results);

    const featureInput = featureExtractor.infer(inputRef.current);
    results = await knnClassifier.classify(featureInput);
    console.log('input ', results);

    console.log(knnClassifier.getCountByLabel());
  };

  const saveModel = () => knnClassifier.save();

  const loadModel = async () => {
    await knnClassifier.load(model);
    console.log('model loaded');
  };

  const loadModelFromInput = async () => {
    console.log(modelUrl);

    await knnClassifier.load(modelUrl);
    console.log('input model loaded');
  };

  const setModel = async file => {
    console.log(file);
    await knnClassifier.load(URL.createObjectURL(file));
    console.log('input model loaded');
    return false; // Preven donit post
  };

  const stopUpload = file => {
    console.log(file);
    return false;
  };

  useEffect(() => {
    const setvars = async () => {
      setKnnClassifier(await ml5.KNNClassifier());
      setFeatureExtractor(await ml5.featureExtractor('MobileNet'));
      console.log('Vars loaded');
    };
    setvars();
  }, []);

  return (
    <div>
      <img
        src={paper1}
        width="224"
        height="224"
        alt=""
        ref={paper1Ref}
        hidden
      />
      <img
        src={paper2}
        width="224"
        height="224"
        alt=""
        ref={paper2Ref}
        hidden
      />
      <img
        src={paper3}
        width="224"
        height="224"
        alt=""
        ref={paper3Ref}
        hidden
      />
      <img src={rock1} width="224" height="224" alt="" ref={rock1Ref} hidden />
      <img src={rock2} width="224" height="224" alt="" ref={rock2Ref} hidden />
      <img src={rock3} width="224" height="224" alt="" ref={rock3Ref} hidden />
      <img src={rock4} width="224" height="224" alt="" ref={rock4Ref} hidden />
      <img
        src={scissors1}
        width="224"
        height="224"
        alt=""
        ref={scissors1Ref}
        hidden
      />
      <img
        src={scissors2}
        width="224"
        height="224"
        alt=""
        ref={scissors2Ref}
        hidden
      />
      <img
        src={scissors3}
        width="224"
        height="224"
        alt=""
        ref={scissors3Ref}
        hidden
      />
      <button onClick={predict}>Predict</button>
      <button onClick={classify}>Train</button>
      <button onClick={saveModel}>Save</button>
      <button onClick={loadModel}>Load</button>
      <button onClick={loadModelFromInput}>inputModel</button>
      <DropZone inputImage={inputRef} />
      <ModelInput setUrl={setModelUrl} />
      <Upload
        name="file"
        beforeUpload={setModel}
        accept=".json, .png"
        onChange={stopUpload}
        action="https://www.mocky.io/v2/5cc8019d300000980a055e76"
        headers={{ authorization: 'authorization-text' }}
      >
        <Button>
          <Icon type="upload" /> Click to Upload
        </Button>
      </Upload>
    </div>
  );
};
