import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the USPTO ID manual dataset
with open('idmanual.json', 'r') as file:
    dataset = json.load(file)

# Preprocess the dataset
descriptions = [entry['id_tx'] for entry in dataset]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(descriptions)

# Build the recommendation model
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(X)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_class():
    if request.method == 'POST':
        data = request.json
        user_input = data['goods_and_services']
        user_vector = vectorizer.transform([user_input])

        # Find the nearest neighbors
        _, indices = model.kneighbors(user_vector)

        # Get the recommended classes
        recommended_classes = [dataset[idx]['Class Name'] for idx in indices[0]]

        return jsonify({'recommended_classes': recommended_classes})
    else:
        return jsonify({
        "id_tx": "009-4140",
        "class_id": "009",
        "description": "Bank note acceptors for separating good bank notes from counterfeits",
        "status": "A"
    },
    {
        "id_tx": "009-4136",
        "class_id": "009",
        "description": "Fingerprint imagers",
        "status": "A"
    },
    {
        "id_tx": "009-4133",
        "class_id": "009",
        "description": "Laboratory swabs [laboratory instruments]",
        "status": "A"
    },
    {
        "id_tx": "009-4131",
        "class_id": "009",
        "description": "Ear plugs for divers",
        "status": "A"
    },
    {
        "id_tx": "009-4130",
        "class_id": "009",
        "description": "DVD recorders",
        "status": "A"
    },
    {
        "id_tx": "009-4129",
        "class_id": "009",
        "description": "Notebook computer carrying cases",
        "status": "A"
    },
    {
        "id_tx": "009-4127",
        "class_id": "009",
        "description": "Ergometers not for medical purposes",
        "status": "A"
    },
    {
        "id_tx": "009-4126",
        "class_id": "009",
        "description": "Scientific apparatus and instruments for measuring relative DNA, RNA and protein and parts and fittings therefor",
        "status": "A"
    },
    {
        "id_tx": "009-4125",
        "class_id": "009",
        "description": "Scientific apparatus, namely, spectrophotometer for measuring relative DNA, RNA and protein",
        "status": "A"
    },
    {
        "id_tx": "009-4124",
        "class_id": "009",
        "description": "Night vision goggles",
        "status": "A"
    },
    {
        "id_tx": "009-4123",
        "class_id": "009",
        "description": "PC tablets",
        "status": "A"
    },
    {
        "id_tx": "009-412",
        "class_id": "009",
        "description": "Personal security alarms",
        "status": "A"
    },
    {
        "id_tx": "009-4119",
        "class_id": "009",
        "description": "Flash card readers",
        "status": "A"
    },
    {
        "id_tx": "009-4118",
        "class_id": "009",
        "description": "Computer docking stations",
        "status": "A"
    },
    {
        "id_tx": "009-4117",
        "class_id": "009",
        "description": "Electronic agendas",
        "status": "A"
    },
    {
        "id_tx": "009-4116",
        "class_id": "009",
        "description": "Wrist rests for use with computers",
        "status": "A"
    },
    {
        "id_tx": "009-4115",
        "class_id": "009",
        "description": "Bullhorns",
        "status": "A"
    },
    {
        "id_tx": "009-4114",
        "class_id": "009",
        "description": "Guitar amplifiers",
        "status": "A"
    },
    {
        "id_tx": "009-4113",
        "class_id": "009",
        "description": "Disposable cameras",
        "status": "A"
    },
    {
        "id_tx": "009-4112",
        "class_id": "009",
        "description": "Cable jump leads",
        "status": "A"
    },
    {
        "id_tx": "009-4111",
        "class_id": "009",
        "description": "Carbon dioxide detectors",
        "status": "A"
    },
    {
        "id_tx": "009-4110",
        "class_id": "009",
        "description": "CD-ROM drives",
        "status": "A"
    },
    {
        "id_tx": "009-4109",
        "class_id": "009",
        "description": "Cordless telephones",
        "status": "A"
    },
    {
        "id_tx": "009-4108",
        "class_id": "009",
        "description": "Radio-frequency controlled locks",
        "status": "A"
    },
    {
        "id_tx": "009-4107",
        "class_id": "009",
        "description": "Blank integrated circuit cards [blank smart cards]",
        "status": "A"
    },
    {
        "id_tx": "009-4106",
        "class_id": "009",
        "description": "Computer whiteboards",
        "status": "A"
    },
    {
        "id_tx": "009-4104",
        "class_id": "009",
        "description": "Microchips [computer hardware]",
        "status": "A"
    },
    {
        "id_tx": "009-4103",
        "class_id": "009",
        "description": "Transponders",
        "status": "A"
    },
    {
        "id_tx": "009-417",
        "class_id": "009",
        "description": "Electronic animal confinement systems",
        "status": "A"
    },
    {
        "id_tx": "009-3757",
        "class_id": "009",
        "description": "Power connectors",
        "status": "A"
    },
    {
        "id_tx": "009-3755",
        "class_id": "009",
        "description": "Power adapters",
        "status": "A"
    },
    {
        "id_tx": "009-3754",
        "class_id": "009",
        "description": "Distribution boxes for electrical power",
        "status": "A"
    },
    {
        "id_tx": "021-4224",
        "class_id": "021",
        "description": "Electrical applicators for applying cosmetics to the skin",
        "status": "M"
    },
    {
        "id_tx": "009-3748",
        "class_id": "009",
        "description": "Measuring apparatus, namely, angle finders",
        "status": "A"
    },
    {
        "id_tx": "009-3747",
        "class_id": "009",
        "description": "Visual and audio recordings featuring {indicate subject matter}",
        "status": "A"
    },
    {
        "id_tx": "009-3746",
        "class_id": "009",
        "description": "Laboratory equipment, namely, microarrays",
        "status": "A"
    },
    {
        "id_tx": "009-3744",
        "class_id": "009",
        "description": "Solid state light based emissive imagers",
        "status": "A"
    },
    {
        "id_tx": "009-3743",
        "class_id": "009",
        "description": "Magnetic stud finders",
        "status": "A"
    },
    {
        "id_tx": "009-3742",
        "class_id": "009",
        "description": "High definition multimedia interface cables",
        "status": "A"
    },
    {
        "id_tx": "009-3741",
        "class_id": "009",
        "description": "Compact discs featuring music and spoken word in the form of poetry",
        "status": "A"
    },
    {
        "id_tx": "009-3740",
        "class_id": "009",
        "description": "Acoustic baffles sold as a component of loudspeakers",
        "status": "A"
    },
    {
        "id_tx": "009-374",
        "class_id": "009",
        "description": "Compact disc players [music]",
        "status": "M"
    },
    {
        "id_tx": "009-3739",
        "class_id": "009",
        "description": "Wireless adapters for computers",
        "status": "A"
    },
    {
        "id_tx": "009-3738",
        "class_id": "009",
        "description": "Electronic stud finders",
        "status": "A"
    },
    {
        "id_tx": "009-3737",
        "class_id": "009",
        "description": "Multi-functional computer terminals with payment function",
        "status": "A"
    },
    {
        "id_tx": "009-3736",
        "class_id": "009",
        "description": "Wireless adapters used to link computers to a telecommunications network",
        "status": "A"
    },
    {
        "id_tx": "028-3994",
        "class_id": "028",
        "description": "Electronic gaming machines, namely, devices which accept a wager",
        "status": "M"
    })

if __name__ == '__main__':
    app.run()
