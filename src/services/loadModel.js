const tf = require('@tensorflow/tfjs-node');
async function loadModel() {
    return tf.loadGraphModel('https://storage.googleapis.com/model-mlgc-chistian/model-in-prod/model.json');
}
module.exports = loadModel;