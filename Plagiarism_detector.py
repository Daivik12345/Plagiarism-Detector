from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np  

file_path = 'Downloads/sample3.txt'


with open(file_path, 'r') as file1:
    
    content1 = file1.read()


file_path2 = 'Downloads/sample2.txt'


with open(file_path2, 'r') as file2:
    
    content2 = file2.read()   
    
combined_content = [content1, content2]


tfidf_vectorizer = TfidfVectorizer()


tfidf_matrix = tfidf_vectorizer.fit_transform(combined_content)


tfidf_array = tfidf_matrix.toarray()


vector1 = tfidf_array[0]
vector2 = tfidf_array[1]


print(len(vector1))
print(len(vector2))

dot_product = np.dot(vector1, vector2)
norm_vector1 = np.linalg.norm(vector1)
norm_vector2 = np.linalg.norm(vector2)
print(dot_product / (norm_vector1 * norm_vector2))