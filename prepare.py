# Functions Defined in Data Preparation Exercises

from sklearn.model_selection import train_test_split

def clean_iris(iris):
    iris = iris.drop(columns = ['species_id', 'measurement_id'])
    iris.rename(columns={'species_name':'species'}, inplace=True)
    dummies = pd.get_dummies(iris['species'])
    iris = pd.concat([iris, dummies], axis=1)
    return iris
def split_iris(iris):
    train, test = train_test_split(iris, test_size=0.2, random_state=123, stratify=iris.species)
    train, validate = train_test_split(train, test_size=0.3, random_state=123, stratify=train.species)
    return train, validate, test
def prep_iris(iris):
    iris = clean_iris(iris)
    train, validate, test = split_iris(iris)
    return train, validate, test

def clean_titanic(titanic):
    titanic = titanic.drop(columns=['passenger_id', 'pclass', 'embarked'])
    titanic = titanic.drop_duplicates()
    dummies = pd.get_dummies(titanic[['class', 'embark_town', 'sex']], drop_first=True)
    titanic = pd.concat([titanic, dummies], axis=1)
    return titanic
def split_titanic(titanic):
    train, test = train_test_split(titanic, test_size = 0.2, random_state=123, stratify=titanic.survived)
    train, validate = train_test_split(train, test_size=0.3, random_state=123, stratify=train.survived)
    return train, validate, test
def prep_titanic(titanic):
    titanic = clean_titanic(titanic)
    train, validate, test = split_titanic(titanic)
    return train, validate, test

def clean_telco(telco):
    telco = telco.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    telco = telco.drop_duplicates()
    dummies = pd.get_dummies(telco[['gender', 'contract_type', 'internet_service_type', 'payment_type']], drop_first=True)
    telco = pd.concat([telco, dummies], axis=1)
    return telco
def split_telco(telco):
    train, test = train_test_split(telco, test_size=0.2, random_state=123, stratify=telco.churn)
    train, validate = train_test_split(train, test_size=0.3, random_state=123, stratify=train.churn)
    return train, validate, test
def prep_telco(telco):
    telco = clean_telco(telco)
    train, validate, test = split_telco(telco)
    return train, validate, test