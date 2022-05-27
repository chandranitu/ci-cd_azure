# load dataset
#pip3 install pycaret
#pip3 uninstall logging



from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models
insurance = get_data('insurance')
print("Number of lines present:-",len(data))

# init environment
from pycaret.regression import *
r1 = setup(insurance, target = 'charges', session_id = 123, 
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True,
           bin_numeric_features= ['age', 'bmi'])

# train a model
lr = create_model('lr')
plot_model(lr)

# save pipeline/model
save_model(lr,'/home/hadoop123/ci-cd_azure/deployment-azure/deployment_28042022.pkl')
