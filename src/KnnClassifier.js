import React, { useEffect, useState, useRef } from 'react';
import * as ml5 from 'ml5';

import paper1 from './images/paper/1yKjzquSvl9ShK7K.png';

import rock1 from './images/rock/0bioBZYFCXqJIulm.png';
import rock2 from './images/rock/0NDYNEoDui7o64gU.png';
import rock3 from './images/rock/2NmrcDGkc7FQuu12.png';
import rock4 from './images/rock/5yHTRIIDcdrXqMYJ.png';

import scissors1 from './images/scissors/0CSaM2vL2cWX6Cay.png';
import scissors2 from './images/scissors/0ePX1wuCc3et7leL.png';
import scissors3 from './images/scissors/1CXgK9fgGdSRggD9.png';

export default () => {
  const [knnClassifier, setKnnClassifier] = useState(null);
  const [featureExtractor, setFeatureExtractor] = useState(null);
  const paper1Ref = useRef(null);
  const rock1Ref = useRef(null);
  const rock2Ref = useRef(null);
  const rock3Ref = useRef(null);
  const rock4Ref = useRef(null);
  const scissors1Ref = useRef(null);
  const scissors2Ref = useRef(null);
  const scissors3Ref = useRef(null);

  let img = new Image(224, 224);

  const classify = async () => {
    // TRAIN IMGS
    // Add examples with a label to classifier
    let features = await featureExtractor.infer(paper1Ref.current);
    knnClassifier.addExample(features, 'paper');

    features = await featureExtractor.infer(rock1Ref.current);
    knnClassifier.addExample(features, 'rock');

    features = await featureExtractor.infer(rock2Ref.current);
    knnClassifier.addExample(features, 'rock');

    features = await featureExtractor.infer(rock3Ref.current);
    knnClassifier.addExample(features, 'rock');

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
      {/* <button onClick={saveModel}>Save</button> */}
      {/* <button onClick={loadModel}>Load</button> */}
    </div>
  );
};
