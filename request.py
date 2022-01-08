import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Angka Melek Huruf Penduduk Umur 15-24 Tahun Menurut Provinsi ':99, 'Angka Partisipasi Sekolah ( A P S )':82})

print(r.json())
