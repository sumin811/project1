from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier



def parsing(path):
    with open(path,'r',encoding='utf-8') as f:
        train=[]
        para=""
        while True:
            l = f.readline()

            
            if l[:6]=='Pragma':
                continue

            if l[:10]=='User-Agent':
                continue

            if l[:10]=='Connection':
                continue

            if not l:
                break 

            if l != "\n":
                para +=l
            else:
                if para!='':
                    if para[:4]=='POST': 
                        para+=f.readline()
                    train.append(para)
                    para=""
    return train

def dataset(path,mod='train'):
    x = parsing(f'{path}norm_{mod}.txt')
    y = [0]*len(x)
    x += parsing(f'{path}anomal_{mod}.txt')
    y += [1]*(len(x)-len(y))
    return x, y



def vectorize(train_x,test_x):
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)
    return train_vec,test_vec

def train(train_vec,train_y):
    kn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    kn.fit(train_vec,train_y)
    return kn

def test(test_y,test_vec,kn):
    pred = kn.predict(test_vec)
    print(accuracy_score(test_y,pred))
    print(f1_score(test_y,pred))
    return pred


def run():
    train_x, train_y = dataset('./','train')
    test_x, test_y =  dataset('./','test')

    train_vec, test_vec = vectorize(train_x, test_x)
    kn = train(train_vec, train_y)
    test(test_y,test_vec, kn)




if __name__=="__main__":
    run()   
