"""
Cloud OCR Services - Detaylı Örnekler
=====================================

Bu dosya, bulut tabanlı OCR servislerinin kullanımına dair 
detaylı örnekler ve konfigürasyon rehberi içermektedir.

Not: Bu servisleri kullanmak için ilgili cloud provider'dan
API anahtarları almanız gerekir.
"""

import os
import json
import base64
import requests
from typing import Dict, List, Any
import time

# ============================================================================
# AWS TEXTRACT DETAYLI ÖRNEKLER
# ============================================================================

class AWSTextractAdvanced:
    """Gelişmiş AWS Textract örnekleri"""
    
    def __init__(self, region_name='us-east-1'):
        """
        AWS Textract kurulumu:
        1. pip install boto3
        2. aws configure (Access Key ID, Secret Access Key, Region ayarlayın)
        """
        try:
            import boto3
            self.textract = boto3.client('textract', region_name=region_name)
            self.s3 = boto3.client('s3', region_name=region_name)
        except ImportError:
            raise ImportError("boto3 kurulu değil. Kurulum: pip install boto3")
    
    def analyze_document_with_tables(self, image_path: str) -> Dict[str, Any]:
        """
        Belge analizi - metin, tablo ve form verileri
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Analiz sonuçları
        """
        try:
            with open(image_path, 'rb') as image:
                img_bytes = image.read()
            
            # Gelişmiş analiz (tablolar ve formlar dahil)
            response = self.textract.analyze_document(
                Document={'Bytes': img_bytes},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            result = {
                'raw_text': '',
                'tables': [],
                'key_value_pairs': [],
                'confidence_stats': {}
            }
            
            # Ham metni çıkar
            text_blocks = []
            confidences = []
            
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text_blocks.append(block['Text'])
                    confidences.append(block['Confidence'])
                
                elif block['BlockType'] == 'KEY_VALUE_SET':
                    # Form alanları
                    if 'KEY' in block.get('EntityTypes', []):
                        key_text = self._extract_text_from_block(block, response['Blocks'])
                        # İlişkili VALUE bloğunu bul
                        for relationship in block.get('Relationships', []):
                            if relationship['Type'] == 'VALUE':
                                value_block_id = relationship['Ids'][0]
                                value_block = next((b for b in response['Blocks'] if b['Id'] == value_block_id), None)
                                if value_block:
                                    value_text = self._extract_text_from_block(value_block, response['Blocks'])
                                    result['key_value_pairs'].append({
                                        'key': key_text,
                                        'value': value_text,
                                        'confidence': block['Confidence']
                                    })
            
            result['raw_text'] = '\n'.join(text_blocks)
            result['confidence_stats'] = {
                'average': sum(confidences) / len(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            }
            
            return result
        
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_text_from_block(self, block: Dict, all_blocks: List[Dict]) -> str:
        """Bloktan metin çıkarır"""
        text_parts = []
        
        for relationship in block.get('Relationships', []):
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in all_blocks if b['Id'] == child_id), None)
                    if child_block and child_block['BlockType'] == 'WORD':
                        text_parts.append(child_block['Text'])
        
        return ' '.join(text_parts)
    
    def process_multipage_pdf_from_s3(self, bucket_name: str, object_key: str) -> Dict[str, Any]:
        """
        S3'teki çok sayfalı PDF'i işler
        
        Args:
            bucket_name: S3 bucket adı
            object_key: Dosya yolu
        
        Returns:
            İşlem sonuçları
        """
        try:
            # Async job başlat
            response = self.textract.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': object_key
                    }
                }
            )
            
            job_id = response['JobId']
            print(f"📄 İş başlatıldı: {job_id}")
            
            # İşin tamamlanmasını bekle
            while True:
                response = self.textract.get_document_text_detection(JobId=job_id)
                status = response['JobStatus']
                
                if status == 'SUCCEEDED':
                    break
                elif status == 'FAILED':
                    return {'error': 'İş başarısız oldu'}
                else:
                    print(f"⏳ Durum: {status}, bekleniyor...")
                    time.sleep(5)
            
            # Sonuçları topla
            pages = {}
            next_token = None
            
            while True:
                if next_token:
                    response = self.textract.get_document_text_detection(
                        JobId=job_id,
                        NextToken=next_token
                    )
                
                for block in response['Blocks']:
                    if block['BlockType'] == 'LINE':
                        page_num = block['Page']
                        if page_num not in pages:
                            pages[page_num] = []
                        pages[page_num].append(block['Text'])
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            return {
                'total_pages': len(pages),
                'pages': {f'page_{k}': '\n'.join(v) for k, v in pages.items()}
            }
        
        except Exception as e:
            return {'error': str(e)}

# ============================================================================
# GOOGLE CLOUD VISION DETAYLI ÖRNEKLER
# ============================================================================

class GoogleVisionAdvanced:
    """Gelişmiş Google Cloud Vision örnekleri"""
    
    def __init__(self, credentials_path: str = None):
        """
        Google Cloud Vision kurulumu:
        1. pip install google-cloud-vision
        2. Service account key indirin
        3. GOOGLE_APPLICATION_CREDENTIALS environment variable ayarlayın
        """
        try:
            from google.cloud import vision
            
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            self.client = vision.ImageAnnotatorClient()
        except ImportError:
            raise ImportError("google-cloud-vision kurulu değil.")
    
    def comprehensive_document_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Kapsamlı belge analizi
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Detaylı analiz sonuçları
        """
        try:
            from google.cloud import vision
            
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Çoklu özellik analizi
            features = [
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
            ]
            
            request = vision.AnnotateImageRequest(image=image, features=features)
            response = self.client.annotate_image(request=request)
            
            result = {
                'full_text': '',
                'text_blocks': [],
                'logos': [],
                'objects': [],
                'languages': []
            }
            
            # Tam metin
            if response.full_text_annotation:
                result['full_text'] = response.full_text_annotation.text
                
                # Dil tespiti
                for page in response.full_text_annotation.pages:
                    for prop in page.property.detected_languages:
                        result['languages'].append({
                            'language': prop.language_code,
                            'confidence': prop.confidence
                        })
            
            # Metin blokları
            for annotation in response.text_annotations[1:]:  # İlkini atla (tam metin)
                vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
                result['text_blocks'].append({
                    'text': annotation.description,
                    'bounding_box': vertices,
                    'confidence': getattr(annotation, 'confidence', 0)
                })
            
            # Logo tespiti
            for logo in response.logo_annotations:
                vertices = [(vertex.x, vertex.y) for vertex in logo.bounding_poly.vertices]
                result['logos'].append({
                    'description': logo.description,
                    'score': logo.score,
                    'bounding_box': vertices
                })
            
            # Nesne tespiti
            for obj in response.localized_object_annotations:
                vertices = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                result['objects'].append({
                    'name': obj.name,
                    'score': obj.score,
                    'bounding_box': vertices
                })
            
            return result
        
        except Exception as e:
            return {'error': str(e)}
    
    def batch_annotate_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Toplu görüntü analizi
        
        Args:
            image_paths: Görüntü dosya yolları listesi
        
        Returns:
            Her görüntü için analiz sonuçları
        """
        try:
            from google.cloud import vision
            
            requests = []
            
            for image_path in image_paths:
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                
                image = vision.Image(content=content)
                feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
                request = vision.AnnotateImageRequest(image=image, features=[feature])
                requests.append(request)
            
            # Toplu istek gönder
            response = self.client.batch_annotate_images(requests=requests)
            
            results = []
            for i, annotation in enumerate(response.responses):
                if annotation.text_annotations:
                    text = annotation.text_annotations[0].description
                else:
                    text = "Metin bulunamadı"
                
                results.append({
                    'image_path': image_paths[i],
                    'text': text,
                    'status': 'success' if not annotation.error.message else 'error',
                    'error': annotation.error.message if annotation.error.message else None
                })
            
            return results
        
        except Exception as e:
            return [{'error': str(e)}]

# ============================================================================
# AZURE COMPUTER VISION DETAYLI ÖRNEKLER
# ============================================================================

class AzureComputerVisionAdvanced:
    """Gelişmiş Azure Computer Vision örnekleri"""
    
    def __init__(self, endpoint: str, subscription_key: str):
        """
        Azure Computer Vision kurulumu:
        1. pip install azure-cognitiveservices-vision-computervision
        2. Azure Portal'dan subscription key alın
        """
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
            
            self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
        except ImportError:
            raise ImportError("azure-cognitiveservices-vision-computervision kurulu değil.")
    
    def read_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Gelişmiş metin okuma (Read API)
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Okuma sonuçları
        """
        try:
            with open(image_path, 'rb') as image_stream:
                # Read API çağrısı
                read_response = self.client.read_in_stream(image_stream, raw=True)
            
            # Operation ID'yi al
            operation_id = read_response.headers['Operation-Location'].split('/')[-1]
            
            # İşlemin tamamlanmasını bekle
            while True:
                read_result = self.client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)
            
            result = {
                'status': read_result.status,
                'text': '',
                'lines': [],
                'words': []
            }
            
            if read_result.status == 'succeeded':
                text_lines = []
                
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        text_lines.append(line.text)
                        
                        # Satır bilgileri
                        bbox = [point for point in line.bounding_box]
                        result['lines'].append({
                            'text': line.text,
                            'bounding_box': bbox,
                            'confidence': getattr(line, 'confidence', 0)
                        })
                        
                        # Kelime bilgileri
                        for word in line.words:
                            word_bbox = [point for point in word.bounding_box]
                            result['words'].append({
                                'text': word.text,
                                'bounding_box': word_bbox,
                                'confidence': word.confidence
                            })
                
                result['text'] = '\n'.join(text_lines)
            
            return result
        
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_document_layout(self, image_path: str) -> Dict[str, Any]:
        """
        Belge layout analizi
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Layout analiz sonuçları
        """
        try:
            with open(image_path, 'rb') as image_stream:
                # Layout analizi
                layout_response = self.client.analyze_layout_in_stream(image_stream, raw=True)
            
            # Operation ID'yi al
            operation_id = layout_response.headers['Operation-Location'].split('/')[-1]
            
            # İşlemin tamamlanmasını bekle
            while True:
                layout_result = self.client.get_analyze_layout_result(operation_id)
                if layout_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)
            
            result = {
                'status': layout_result.status,
                'pages': [],
                'tables': [],
                'paragraphs': []
            }
            
            if layout_result.status == 'succeeded':
                for page in layout_result.analyze_result.page_results:
                    page_info = {
                        'page_number': page.page,
                        'width': page.width,
                        'height': page.height,
                        'unit': page.unit,
                        'tables': []
                    }
                    
                    # Tablolar
                    if hasattr(page, 'tables'):
                        for table in page.tables:
                            table_data = {
                                'rows': table.rows,
                                'columns': table.columns,
                                'cells': []
                            }
                            
                            for cell in table.cells:
                                table_data['cells'].append({
                                    'text': cell.text,
                                    'row_index': cell.row_index,
                                    'column_index': cell.column_index,
                                    'row_span': cell.row_span,
                                    'column_span': cell.column_span,
                                    'confidence': cell.confidence
                                })
                            
                            page_info['tables'].append(table_data)
                    
                    result['pages'].append(page_info)
            
            return result
        
        except Exception as e:
            return {'error': str(e)}

# ============================================================================
# BULUT SERVİSLERİ KARŞILAŞTIRMA
# ============================================================================

def compare_cloud_ocr_services(image_path: str, 
                               aws_region: str = 'us-east-1',
                               google_credentials: str = None,
                               azure_endpoint: str = None,
                               azure_key: str = None) -> Dict[str, Any]:
    """
    Bulut OCR servislerini karşılaştırır
    
    Args:
        image_path: Test görüntüsü
        aws_region: AWS bölgesi
        google_credentials: Google service account key path
        azure_endpoint: Azure endpoint URL
        azure_key: Azure subscription key
    
    Returns:
        Karşılaştırma sonuçları
    """
    results = {}
    
    # AWS Textract
    try:
        aws_ocr = AWSTextractAdvanced(aws_region)
        start_time = time.time()
        aws_result = aws_ocr.analyze_document_with_tables(image_path)
        aws_time = time.time() - start_time
        
        results['aws_textract'] = {
            'text': aws_result.get('raw_text', ''),
            'time': aws_time,
            'features': ['tables', 'forms', 'key_value_pairs'],
            'confidence': aws_result.get('confidence_stats', {}),
            'status': 'success' if 'error' not in aws_result else 'error'
        }
    except Exception as e:
        results['aws_textract'] = {'error': str(e), 'status': 'error'}
    
    # Google Cloud Vision
    try:
        google_ocr = GoogleVisionAdvanced(google_credentials)
        start_time = time.time()
        google_result = google_ocr.comprehensive_document_analysis(image_path)
        google_time = time.time() - start_time
        
        results['google_vision'] = {
            'text': google_result.get('full_text', ''),
            'time': google_time,
            'features': ['text_detection', 'logo_detection', 'object_localization'],
            'languages': google_result.get('languages', []),
            'status': 'success' if 'error' not in google_result else 'error'
        }
    except Exception as e:
        results['google_vision'] = {'error': str(e), 'status': 'error'}
    
    # Azure Computer Vision
    if azure_endpoint and azure_key:
        try:
            azure_ocr = AzureComputerVisionAdvanced(azure_endpoint, azure_key)
            start_time = time.time()
            azure_result = azure_ocr.read_text_from_image(image_path)
            azure_time = time.time() - start_time
            
            results['azure_cv'] = {
                'text': azure_result.get('text', ''),
                'time': azure_time,
                'features': ['read_api', 'layout_analysis'],
                'words_count': len(azure_result.get('words', [])),
                'status': azure_result.get('status', 'error')
            }
        except Exception as e:
            results['azure_cv'] = {'error': str(e), 'status': 'error'}
    
    return results

# ============================================================================
# KONFİGÜRASYON REHBERİ
# ============================================================================

def setup_guide():
    """Bulut servisleri kurulum rehberi"""
    
    guide = """
    🌤️  BULUT OCR SERVİSLERİ KURULUM REHBERİ
    ==========================================
    
    1️⃣  AWS TEXTRACT KURULUMU:
    ---------------------------
    
    a) AWS CLI kurulumu:
       curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
       sudo installer -pkg AWSCLIV2.pkg -target /
    
    b) AWS Credentials ayarlama:
       aws configure
       # Access Key ID girin
       # Secret Access Key girin
       # Default region (us-east-1) girin
    
    c) Python paketi:
       pip install boto3
    
    d) Test:
       aws sts get-caller-identity
    
    
    2️⃣  GOOGLE CLOUD VISION KURULUMU:
    ----------------------------------
    
    a) Service Account oluşturma:
       - Google Cloud Console'a gidin
       - IAM & Admin > Service Accounts
       - "Create Service Account" tıklayın
       - Vision API izinleri verin
       - JSON key indirin
    
    b) Environment variable ayarlama:
       export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
    
    c) Python paketi:
       pip install google-cloud-vision
    
    d) Test:
       gcloud auth application-default print-access-token
    
    
    3️⃣  AZURE COMPUTER VISION KURULUMU:
    ------------------------------------
    
    a) Azure Portal'dan resource oluşturma:
       - portal.azure.com'a gidin
       - "Create a resource" > "Computer Vision"
       - Resource grup seçin/oluşturun
       - Pricing tier seçin (F0 ücretsiz)
    
    b) Credentials alma:
       - Resource'a gidin
       - "Keys and Endpoint" seçin
       - Key ve Endpoint kopyalayın
    
    c) Python paketi:
       pip install azure-cognitiveservices-vision-computervision
    
    d) Environment variables:
       export AZURE_CV_ENDPOINT="your-endpoint"
       export AZURE_CV_KEY="your-key"
    
    
    💡 İPUÇLARI:
    =============
    
    • Ücretsiz tier'lar sınırlıdır, production için ücretli planlara geçin
    • API rate limit'lerine dikkat edin
    • Büyük dosyalar için async processing kullanın
    • Güvenlik için credentials'ı environment variables'da saklayın
    • Her servisin fiyatlandırmasını karşılaştırın
    
    
    📊 PERFORMANS KARŞILAŞTIRMASI:
    ==============================
    
    | Servis           | Güçlü Yanları              | Zayıf Yanları           |
    |------------------|-----------------------------|-------------------------|
    | AWS Textract     | Tablolar, formlar          | Daha pahalı            |
    | Google Vision    | Dil desteği, logo tespiti  | Layout analizi zayıf   |
    | Azure CV         | Hızlı, iyi API            | Özellik seti kısıtlı   |
    
    """
    
    return guide

# ============================================================================
# DEMO FUNCTION
# ============================================================================

def demo_cloud_services():
    """Bulut servisleri demo'su"""
    
    print("☁️  Bulut OCR Servisleri Demo'su")
    print("=" * 50)
    
    print(setup_guide())
    
    # Kullanıcıdan credentials istemek yerine, örnek kodları göster
    print("\n📝 ÖRNEK KULLANIM KODLARI:")
    print("-" * 30)
    
    print("""
    # AWS Textract Örneği:
    aws_ocr = AWSTextractAdvanced('us-east-1')
    result = aws_ocr.analyze_document_with_tables('fatura.png')
    print(result['raw_text'])
    
    # Google Vision Örneği:
    google_ocr = GoogleVisionAdvanced('/path/to/service-key.json')
    result = google_ocr.comprehensive_document_analysis('belge.jpg')
    print(result['full_text'])
    
    # Azure CV Örneği:
    azure_ocr = AzureComputerVisionAdvanced(endpoint, key)
    result = azure_ocr.read_text_from_image('kimlik.png')
    print(result['text'])
    
    # Karşılaştırma:
    comparison = compare_cloud_ocr_services('test.jpg')
    for service, data in comparison.items():
        print(f"{service}: {data.get('status', 'error')}")
    """)

if __name__ == "__main__":
    demo_cloud_services()