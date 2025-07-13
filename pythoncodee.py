import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import pickle

# Load dataset (assumes the CSV from previous response is available)
data = pd.read_csv('sustainability_dataset_300_with_dimensions.csv')

# Function to parse dimensions
def parse_dimensions(dimensions):
    try:
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', dimensions)]
        if len(nums) == 1:
            length = nums[0]
            area = length
            shape = 'linear'
            width = height = 0
        else:
            length, width, height = nums[:3]
            area = length * width
            shape = 'volumetric' if height > 2 else 'flat'
        size = 'small' if area < 200 or (shape == 'linear' and length < 15) else 'medium' if area < 1000 or (shape == 'linear' and length < 30) else 'large'
        return shape, size, length, width, height
    except:
        return 'unknown', 'medium', 0, 0, 0

# Rule-based Green Score for training (same as previous)
def calculate_green_score(row):
    score = 0
    max_score = 100

    # Material Analysis (35 points)
    material_keywords = {
        'bamboo': 15, 'bagasse': 15, 'plant-based': 15, 'recycled': 10,
        'plastic': -10, 'non-recyclable': -15
    }
    description = row['description'].lower()
    for material, points in material_keywords.items():
        if material in description:
            score += points

    # Sustainability Labels (30 points)
    label_points = {
        'biodegradable': 10, 'compostable': 10, 'organic': 8,
        'plastic-free': 8, 'vegan': 5, 'ethical': 5, 'recyclable': 5
    }
    labels = row['sustainability_labels'].lower().split(',')
    for label in labels:
        label = label.strip()
        score += label_points.get(label, 0)

    # Brand Reputation (20 points)
    eco_brands = ['EcoBrush', 'GreenTable', 'NaturePure', 'GreenLeaf', 'EcoWear', 'SteelSip', 'EcoBrew', 'GreenFork', 'PureBloom', 'ClearCycle', 'ZenMat', 'CleanWave', 'EcoRest', 'GreenCarry', 'EcoBin', 'TinyGreen', 'EcoTech', 'SeaShade', 'PureLeaf', 'EcoSip', 'GreenComb', 'EcoFeet', 'CleanGreen', 'EcoCard', 'FreshEco', 'GreenDine', 'EcoDesk', 'GreenPack', 'PureBlend', 'EcoStore', 'SmileGreen', 'CycleWorks', 'GlowGreen', 'EcoShade', 'GreenChef']
    if row['brand'] in eco_brands:
        score += 20

    # Price-to-Sustainability Ratio (10 points)
    if row['price'] < 10 and any(label in row['sustainability_labels'].lower() for label in label_points):
        score += 10

    # Dimensions Impact (5 points)
    shape, size, _, _, _ = parse_dimensions(row['dimensions'])
    if size == 'small' or shape == 'linear':
        score += 5

    score = max(0, min(score, max_score))
    grade = 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D' if score >= 20 else 'E'
    return score, grade

# Generate Green Scores for training
data['green_score'], data['green_grade'] = zip(*data.apply(calculate_green_score, axis=1))

# Feature Engineering
# Combine description and sustainability labels for TF-IDF
data['text'] = data['description'] + ' ' + data['sustainability_labels']
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(data['text']).toarray()

# Extract dimension features
dimension_features = data['dimensions'].apply(parse_dimensions)
data['shape'], data['size'], data['length'], data['width'], data['height'] = zip(*dimension_features)

# Numerical features: price, length, width, height
num_features = data[['price', 'length', 'width', 'height']].values

# Combine text and numerical features
X = np.hstack((text_features, num_features))
y = data['green_score'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Save model and vectorizer
with open('green_score_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# DIY Reuse Ideas Based on Dimensions
def generate_diy_ideas(dimensions):
    shape, size, length, _, _ = parse_dimensions(dimensions)
    
    product_reuse = []
    if shape == 'linear' and size == 'small':
        product_reuse.append("Use as a plant marker by sticking it into soil to label herbs or seedlings.")
        product_reuse.append("Repurpose as a small stirrer for paints or crafts.")
    elif shape == 'linear' and size in ['medium', 'large']:
        product_reuse.append("Use as a garden stake to support small plants or vines.")
        product_reuse.append("Repurpose as a tool handle for DIY projects like sculpting or cleaning.")
    elif shape == 'flat' and size == 'small':
        product_reuse.append("Use as a coaster for drinks to protect surfaces.")
        product_reuse.append("Repurpose as a base for small craft projects or mosaics.")
    elif shape == 'flat' and size in ['medium', 'large']:
        product_reuse.append("Use as a reusable placemat for picnics or outdoor dining.")
        product_reuse.append("Repurpose as a fabric for DIY tote bags or wall hangings.")
    elif shape == 'volumetric' and size == 'small':
        product_reuse.append("Use as a small storage container for desk supplies or jewelry.")
        product_reuse.append("Repurpose as a mini planter for succulents or herbs.")
    elif shape == 'volumetric' and size in ['medium', 'large']:
        product_reuse.append("Use as a storage box for household items or toys.")
        product_reuse.append("Repurpose as a base for a DIY lamp or decorative centerpiece.")

    packaging_reuse = []
    if shape == 'linear' and size == 'small':
        packaging_reuse.append("Use the small packaging as a holder for pens or small tools.")
        packaging_reuse.append("Shred the packaging to use as cushioning for future shipments.")
    elif shape == 'linear' and size in ['medium', 'large']:
        packaging_reuse.append("Use the packaging as a protective sleeve for tools or delicate items.")
        packaging_reuse.append("Repurpose as a seedling tray for gardening.")
    elif shape == 'flat' and size == 'small':
        packaging_reuse.append("Use the packaging as a base for DIY greeting cards or bookmarks.")
        packaging_reuse.append("Repurpose as wrapping paper for small gifts.")
    elif shape == 'flat' and size in ['medium', 'large']:
        packaging_reuse.append("Use the packaging as a reusable gift wrap for larger items.")
        packaging_reuse.append("Repurpose as a protective mat for crafting or painting.")
    elif shape == 'volumetric' and size == 'small':
        packaging_reuse.append("Use the packaging as a small organizer for desk or bathroom items.")
        packaging_reuse.append("Repurpose as a seedling starter pot for gardening.")
    elif shape == 'volumetric' and size in ['medium', 'large']:
        packaging_reuse.append("Use the packaging as a storage box for household items or clothes.")
        packaging_reuse.append("Repurpose as a base for a DIY storage ottoman or decorative box.")

    return product_reuse, packaging_reuse

# Function to predict Green Score and generate DIY ideas for a new product
def predict_and_generate_ideas(product_data):
    # product_data: dict with keys: product_name, description, brand, category, sustainability_labels, price, dimensions
    text = product_data['description'] + ' ' + product_data['sustainability_labels']
    text_features = tfidf.transform([text]).toarray()
    
    shape, size, length, width, height = parse_dimensions(product_data['dimensions'])
    num_features = np.array([[product_data['price'], length, width, height]])
    
    X_new = np.hstack((text_features, num_features))
    green_score = model.predict(X_new)[0]
    green_grade = 'A' if green_score >= 80 else 'B' if green_score >= 60 else 'C' if green_score >= 40 else 'D' if green_score >= 20 else 'E'
    
    product_reuse, packaging_reuse = generate_diy_ideas(product_data['dimensions'])
    
    return {
        'product_name': product_data['product_name'],
        'green_score': round(green_score, 2),
        'green_grade': green_grade,
        'product_reuse_ideas': product_reuse,
        'packaging_reuse_ideas': packaging_reuse,
        'dimensions': product_data['dimensions']
    }

# Example: Predict for a new product
new_product = {
    'product_name': 'Reusable Bamboo Straw',
    'description': 'A durable straw made from sustainable bamboo, eco-friendly.',
    'brand': 'EcoSip',
    'category': 'Accessories',
    'sustainability_labels': 'biodegradable,reusable',
    'price': 3.99,
    'dimensions': '20 cm'
}

result = predict_and_generate_ideas(new_product)
print(f"Product: {result['product_name']}")
print(f"Dimensions: {result['dimensions']}")
print(f"Green Score: {result['green_score']} ({result['green_grade']})")
print("Product Reuse Ideas:")
for idea in result['product_reuse_ideas']:
    print(f"- {idea}")
print("Packaging Reuse Ideas:")
for idea in result['packaging_reuse_ideas']:
    print(f"- {idea}")

# Example Output for a few products from the dataset
def process_dataset(df):
    result_list = []

    for _, row in df.iterrows():
        product = {
            'product_name': row['product_name'],
            'description': row['description'],
            'brand': row['brand'],
            'category': row['category'],
            'sustainability_labels': row['sustainability_labels'],
            'price': row['price'],
            'dimensions': row['dimensions']
        }
        result = predict_and_generate_ideas(product)
        result_list.append(result)

    return pd.DataFrame(result_list)

results = process_dataset(data.head(3))
for _, row in results.iterrows():
    print(f"\nProduct: {row['product_name']}")
    print(f"Dimensions: {row['dimensions']}")
    print(f"Green Score: {row['green_score']} ({row['green_grade']})")
    print("Product Reuse Ideas:")
    for idea in row['product_reuse_ideas']:
        print(f"- {idea}")
    print("Packaging Reuse Ideas:")
    for idea in row['packaging_reuse_ideas']:
        print(f"- {idea}")