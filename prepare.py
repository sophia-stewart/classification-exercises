# Functions Defined in Data Preparation Exercises

def prep_iris(iris):
    iris = iris.drop(columns = ['species_id', 'measurement_id'])
    iris.rename(columns={'species_name':'species'}, inplace=True)
    dummies = pd.get_dummies(iris['species'])
    iris = pd.concat([iris, dummies], axis=1)
    return iris

def prep_titanic(titanic):
    titanic = titanic.drop(columns=['passenger_id', 'pclass', 'embarked'])
    dummies = pd.get_dummies(titanic[['class', 'embark_town', 'sex']], drop_first=True)
    titanic = pd.concat([titanic, dummies], axis=1)
    return titanic

def prep_telco(telco):
    telco = telco.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    dummies = pd.get_dummies(telco[['gender', 'contract_type', 'internet_service_type', 'payment_type']], drop_first=True)
    telco = pd.concat([telco, dummies], axis=1)
    return telco