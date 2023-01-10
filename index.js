
const filesElement = document.getElementById('files');

let reads = [];

filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  //console.log(files[0]);
  //let reads = [];

  for (let i = 0, f; f = files[i]; i++) {
	//if (!f.type.match('text.*')) {
    //  continue;
    //}
    let reader = new FileReader();
    reader.onload = e => {
	  reads =  reader.result.trim().split("\n");
	  main(reads);
    };

    // Read read sequences as text
    reader.readAsText(f);
    //reader.readAsDataURL(f);
  }
});


async function main(reads) {
  console.time("Inference");

  let modelDict = {
    'colorectal-cancer-diagnosis': 'models/CRC_diagnosis/attn_based_deep_mil_crc_dx.onnx',
    'cancer-type-classification': 'models/tissue_of_origin/attn_based_deep_mil_tissue_of_origin.onnx',
  };

  const featureExtractor = await ort.InferenceSession.create("models/feature_extractor/clm-2m/pytorch_model.onnx");
  const classifier = await ort.InferenceSession.create(modelDict[ClassifierID.value]);

  const base_dict = {"A":4n, "C":6n, "G":10n, "T":23n, "N":17n};
  const START_TOKEN_ID = 2n;
  const SEQ_LENGTH = 20;

  let features = [];
  let counter = 0;

  for (var i = 0; i < reads.length; i++) {
    let s = reads[i].trim().toUpperCase();
    if (s.length < SEQ_LENGTH) continue;
    if (s.length > SEQ_LENGTH) {
      s = s.substring(0, SEQ_LENGTH);
    }
    var input_ids = [START_TOKEN_ID];
    for (var j = 0; j < SEQ_LENGTH; j++) {
      const b = s.charAt(j);
      input_ids.push(base_dict[b])
    }
    const ids = new ort.Tensor("int64", input_ids, [1, input_ids.length]);
    const results = await featureExtractor.run({input_ids: ids});

    const feature = await featureExtractor.run({input_ids: ids});
    features = [...features, ...feature.last_hidden_state_mean.data];

    counter += 1;
  }

  const featureTensors = new ort.Tensor("float32", features, [counter, 384]);
  const results = await classifier.run({inputs_embeds: featureTensors});

  if (ClassifierID.value == 'colorectal-cancer-diagnosis') {
    setCRCText(results);
  } else {
	setCancerTypeText(results);
  }

  console.log(results);
  console.timeEnd("Inference");
}

function setCRCText(results) {
  const EXP1 = Math.exp(results.logits.data[0]);
  const EXP2 = Math.exp(results.logits.data[1]);
  const cancerProb = EXP2 / (EXP1 + EXP2);
  const controlProb = EXP1 / (EXP1 + EXP2);
  document.getElementById('predictions').innerHTML = "CRC: " + cancerProb.toFixed(5) + ", " + "Control: " + controlProb.toFixed(5) + ".";
}

function setCancerTypeText(results) {
  const EXP1 = Math.exp(results.logits.data[0]);
  const EXP2 = Math.exp(results.logits.data[1]);
  const EXP3 = Math.exp(results.logits.data[2]);
  const CRCProb = EXP1 / (EXP1 + EXP2 + EXP3);
  const HCCProb = EXP2 / (EXP1 + EXP2 + EXP3);
  const LungCAProb = EXP3 / (EXP1 + EXP2 + EXP3);
  document.getElementById('predictions').innerHTML = "CRC: "+CRCProb.toFixed(5)+", "+"HCC: "+HCCProb.toFixed(5)+", Lung cancer: "+LungCAProb.toFixed(5)+".";
}

