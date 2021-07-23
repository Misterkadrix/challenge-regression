from utils.a_import_raw_data import preprocess
from utils.d_model_global_all_features import model_global_all_features
from utils.e_multiple_linear import multiple_regression_selected_features
from utils.f_gradient_boosting import gradientboostingregressor
from utils.g_gradient_boosting_apartment import GBR_apartment
from utils.h_gradient_boosting_house import GBR_House


def linear_regression_models_all_results():
    '''
    This function will import the raw data, clean it, and run 4 different linear regression model including multiple
    linear regression and gradient boosting regressor.
    :return: Score of each model.
    '''

    #call the preprocessing function
    ds = preprocess()

    #call the global multiple linear regression and report its score
    global_model_result = model_global_all_features(ds)
    print(global_model_result)

    #call the model of multiple linear regression with selected features and report its score
    result_mul_reg = multiple_regression_selected_features(ds)
    print(result_mul_reg)

    #call the gradient boosting regressor with all numeric features included and report its score
    gradient_boosting_result = gradientboostingregressor(ds)
    print(gradient_boosting_result)

    #Gradient boosting regressor score for house prices
    gradient_regressor_house_result = GBR_House(ds)
    print(gradient_regressor_house_result)

    # Gradient boosting regressor score for apartment prices
    gradient_boosting_regressor_for_apartment_result = GBR_apartment(ds)
    print(gradient_boosting_regressor_for_apartment_result)

linear_regression_models_all_results()
