import joblib
import pickle
import warnings




if 'google.colab' in str(get_ipython()):
    model_dir = "MPEAsAlloyPublic/models"
else:
    model_dir = "models"
    
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=UserWarning)

warnings.filterwarnings('ignore')

models = {"Current density": joblib.load(f"{model_dir}/RF_icorr"),
          "Corrosion potential": joblib.load(f"{model_dir}/RF_Ecorr"),
          "Pitting potential": joblib.load(f"{model_dir}/RF_Pcorr")
          }

