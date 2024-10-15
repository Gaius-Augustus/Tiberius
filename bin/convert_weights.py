import tensorflow as tf
from models import custom_cce_f1_loss
from packaging import version


assert((version.parse(tf.__version__) >= version.parse("2.10.0") and version.parse(tf.__version__) < version.parse("2.11.0") or
        version.parse(tf.__version__) >= version.parse("2.17.0")), 
        "Run this script with tensorflow 2.10.x or 2.17.x. Your TF: " + tf.__version__)


# loads the old Tiberius models and converts them to H5 format, which can be loaded by keras 3 (tf 2.17+)

model_path = "../model_weights/"

# if run under tf 2.10.x; convert to H5
if version.parse(tf.__version__) >= version.parse("2.10.0") and version.parse(tf.__version__) < version.parse("2.11.0"):
    for model_name in ["tiberius_weights", "tiberius_nosm_weights"]:
        model = tf.keras.models.load_model(model_path + model_name, 
                                            custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, 32),
                                                'loss_': custom_cce_f1_loss(2, 32)})
        model.save_weights(model_path + model_name + ".h5")
#else if run under tf 2.17.x; load from H5 and save again
elif version.parse(tf.__version__) >= version.parse("2.17.0"):
    for model_name in ["tiberius_weights", "tiberius_nosm_weights"]:
        pass
        #model = #TODO, create a new model with matching architecture and load weights from the H5 file
        #TODO, save new model
        