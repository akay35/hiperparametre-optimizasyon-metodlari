## Hiperparametre Optimizasyon Metodları ( Hyperparameter Optimization Methods )

### Giriş
Hiperparametreler, makine öğrenimi modellerinin öğrenme sürecini kontrol eden ve doğrudan modelin başarısını etkileyen ayarlardır. 
Yanlış veya varsayılan hiperparametrelerle eğitilen bir model, düşük performans gösterebilir ya da önemli bir genelleme hatasına yol açabilir. 
Bu nedenle, modelin doğruluğunu ve genelleme yeteneğini en üst düzeye çıkarmak için hiperparametrelerin dikkatlice seçilmesi gerekir.

GridSearchCV, RandomizedSearchCV ve daha yeni yaklaşımlar olan Bayesian Optimization, Tree-structured Parzen Estimator (TPE) gibi yöntemler, hiperparametre optimizasyonunu sistematik ve etkili bir şekilde yapılmasına olanak tanır. 
Bu yöntemler, manuel deneme yanılma sürecinin yerini alarak hem zaman kazandırır hem de daha iyi sonuçlar elde etmemizi sağlar. 

---

### 1. RandomizedSearchCV

###### Scikit-learn kütüphanesinde bulunan ve hızlı olması dolayısıyla GridSearchCV'ye alternatif olarak kullanılan bir optimizasyon yöntemidir.
###### RandomizedSearchCV belirli bir sayıda rastgele seçilmiş hiperparametre kombinasyonunu test ederek sorunu çözmeye çalışır.
###### Bu sayede arama süresini kısaltabilir ve daha hızlı sonuçlar elde edilmesine olanak tanır.
###### Özellikle büyük veri kümeleri veya çok sayıda hiperparametre kombinasyonu olan durumlarda kullanışlıdır. 
###### Deneme sayısını belirleyerek (örneğin, n_iter=100), daha esnek ve hızlı bir şekilde en iyi hiperparametrelerin bulunmasına yardımcı olur.

![image](https://github.com/akay35/hiperparametre-optimizasyon-metodlari/blob/main/RandomizedSearchCV.png)

###### n_iter=100 ifadesi, RandomizedSearchCV'nin hiperparametre optimizasyonu sırasında deneyeceği 100 farklı kombinasyonu belirtmektedir. Bu, modelin hiperparametreleri için rastgele seçilen 100 farklı seti test edeceği anlamına gelir.
###### Hiperparametrelerin dağılımları (param_distributions) belirlenir. Örneğin, öğrenme hızı (learning_rate), maksimum derinlik (max_depth), ağaç sayısı (n_estimators) gibi hiperparametreler için belirli aralıklar veya olasılık dağılımları tanımlanır.
###### RandomizedSearchCV, her bir hiperparametre için belirtilen aralıktan veya dağılımdan rastgele değerler seçer. Bu seçilen değerler, modelin hiperparametre kombinasyonlarını oluşturur.
###### n_iter=100 olduğu için, RandomizedSearchCV bu hiperparametrelerin 100 farklı rastgele kombinasyonunu oluşturur ve her birini test eder.
###### Eğer learning_rate için 4 farklı değer, max_depth için 5 farklı değer ve n_estimators için 10 farklı değer varsa, bu kombinasyonlar arasında rastgele seçimler yapılır ve toplamda 100 farklı kombinasyon test edilir.

###### Hızlı olması, yüksek boyutlu veriler için uygun olması ve optimizasyon için pratik olması gibi avantajlarının yanında, tüm olası kombinasyonları denemediği için bazen en iyi hiperparametre setini kaçırabilmesi ve rastgele olarak çalışma yapması dezavantajları olarak sayılabilir.

---

### 2. GridSearchCV
###### Scikit-learn kütüphanesinde bulunan bu yöntem, belirli hiperparametrelerin tüm olası kombinasyonlarını dener ve en iyi sonucu veren parametre setini bulur. 
###### "Grid" kelimesi, parametrelerin farklı değerlerinin oluşturduğu bir ızgaraya (grid) atıfta bulunur, çünkü her hiperparametre için belirli bir dizi değer belirlenir ve bu değerlerin her bir kombinasyonu test edilir.

![image](https://github.com/akay35/hiperparametre-optimizasyon-metodlari/blob/main/GridSearchCV.png)

###### Yukarıdaki resimdeki örnekte GridSearchCV bu 4 x 12 x 5 x 10 = 2400  kombinasyonu sırasıyla test eder.
###### Her parametre kombinasyonu için model yeniden eğitilir ve genellikle k-katlamalı çapraz doğrulama (k-fold cross-validation) ile değerlendirilir. 
###### Çapraz doğrulama, modelin doğruluğunu daha güvenilir bir şekilde ölçmek için veri setini birkaç alt kümeye böler ve her alt kümede modelin performansını test eder.

###### GridSearch tüm kombinasyonları denerken daha kesin sonuçlar sunar, RandomizedSearch ise geniş aralıklar için daha hızlı ve pratiktir.

---

### 3. BayesSearchCV

###### Bu yöntem, özellikle hiperparametrelerin sayısının çok fazla olduğu durumlarda daha verimli ve hızlı sonuçlar almayı sağlar. 
###### BayesSearchCV, klasik yöntemler (örneğin GridSearchCV veya RandomizedSearchCV) ile karşılaştırıldığında daha az parametre setiyle daha iyi sonuçlar elde etmeye çalışır.

- İlk başta, modelin hiperparametreleri için rastgele bazı kombinasyonlar seçilir ve bu kombinasyonların nasıl performans gösterdiği test edilir.
- Seçilen parametre kombinasyonlarının başarısı değerlendirilir ve bu sonuçlar bir modelde kullanılarak gelecekteki denemelere rehberlik eder.
- İlk denemelerden elde edilen sonuçlara dayanarak, BayesSearchCV daha iyi performans gösterme olasılığı olan yeni parametre kombinasyonlarını önerir. Bu öneriler, bir acquisition function (kazanç fonksiyonu) kullanılarak yapılır. Bu fonksiyon, hangi parametrelerin deneneceği konusunda bir strateji geliştirir.
- Önerilen yeni parametre kombinasyonları test edilir ve sonuçlar modelin bir sonraki öneri için daha da iyileştirilmesine yardımcı olur.
- Bu işlem tekrar edilerek, modelin performansı sürekli olarak iyileştirilir. Yani, BayesSearchCV daha hızlı ve verimli bir şekilde en iyi parametre kombinasyonlarını bulur.

![image](https://github.com/akay35/hiperparametre-optimizasyon-metodlari/blob/main/BayesSearchCV.png)

Yukarıdaki örnekte BayesSearchCV parametre ayarlamalarının açıklamaları aşağıdaki gibidir:
- estimator: Hangi modelin optimizasyon yapılacağı belirtilir. Burada lgbm_model (LightGBM sınıflandırıcı) kullanılıyor.
- search_spaces: Parametre arama alanlarını tanımlar. Yani, hangi parametrelerin hangi değer aralıklarında test edileceğini belirleriz. Burada yukarıda tanımladığımız param_grid kullanılmıştır.
- n_iter: Optimizasyon sürecinde kaç iterasyon yapılacağını belirtir. Burada 32 iterasyon yapılacak, yani toplamda 32 farklı parametre kombinasyonunun test edileceğini belirtir.
- cv: 5-fold çapraz doğrulama (cross-validation) kullanılır. Bu, modelin daha sağlam sonuçlar verebilmesi için veriyi 5 eşit parçaya ayırıp her bir parçayı sırayla test ve eğitim için kullanmayı sağlar.
- scoring: Modelin başarısını ölçmek için roc_auc (Receiver Operating Characteristic - Area Under Curve) metriği kullanılmıştır. Bu, sınıflandırma problemleri için yaygın bir başarı ölçütüdür.

---

### 4. HalvingGridSearchCV

HalvingGridSearchCV, GridSearchCV'nin genişletilmiş bir versiyonudur. GridSearchCV, verilen hiperparametre grid'inde (matrisinde) her kombinasyonu test ederek en iyi parametreyi bulmaya çalışır. Ancak, bu çok sayıda parametre ve model eğitimi gerektirebilir ve işlem süresi uzun olabilir. HalvingGridSearchCV, bu süreci hızlandırmak için daha verimli bir yöntem sunar.

##### HalvingGridSearchCV'nin temel mantığı şu şekildedir:
- İlk olarak daha küçük bir alt küme (Daha küçük veri setiyle) kullanılarak parametrelerin doğruluğu hesaplanır.
- İlk adımda iyi performans gösteren parametreler seçilir.
- Yüksek doğruluk göstermeyen parametreler, sonraki adımlarda deneme dışı bırakılır.
- Seçilen parametrelerle model tekrar eğitilir ancak bu kez daha büyük bir veri kümesi kullanılır.
- Sonuçları sırasıyla daraltmak: Her adımda, bir sonraki deneme için yalnızca iyi performans gösteren parametreler kullanılarak daha küçük, ancak daha anlamlı kombinasyonlar üzerinde çalışılır.

##### HalvingGridSearchCV'nin Temel Parametreleri:
![image](https://github.com/akay35/hiperparametre-optimizasyon-metodlari/blob/main/HalvingGridSearchCV.png)

- estimator    : Hangi modelin kullanılacağını belirtir.
- param_grid   : Denenecek parametrelerin bir gridi.
- n_candidates : İlk denemede kaç parametre kombinasyonunun test edileceği.
- factor       : Her adımda kaybeden parametrelerin ne kadar azaltılacağını belirtir (örneğin, 3, 4, 5 gibi). Bu, daha fazla parametreyi hızlıca elemeye yardımcı olur.
- min_resources: İlk denemede kullanılacak en küçük örnek sayısı.
- n_jobs       : Paralel işleme sayısı.

---

### 5. Evolutionary Algorithms

TPOT (Tree-based Pipeline Optimization Tool), evrimsel algoritmalarla otomatik makine öğrenimi (AutoML) sağlayan bir Python kütüphanesidir. TPOT, veriler üzerinde model oluşturma sürecini otomatize eder. Bu, genetik algoritmalar kullanarak farklı model yapılarını (algoritmalar, hiperparametreler, veri işleme yöntemleri vb.) deneyip optimize eder.

##### TPOT ve Evrimsel Algoritmaların Çalışma Prensibi
TPOT, evrimsel algoritmalarla çalışarak çeşitli model yapılarını "evrimleştirir" ve en iyi performansı gösteren model türevini bulur. Temel olarak şu adımları takip eder:

- Başlangıç Popülasyonu: TPOT, başlangıçta farklı makine öğrenimi modellerinin rastgele kombinasyonlarını içerir. Bu modeller, algoritmaların ve hiperparametrelerin birleşiminden oluşur.
- Fitness (Uygunluk) Fonksiyonu: Her bireyin başarısını (fitness'ını) ölçmek için genellikle modelin doğruluk oranı, f1 skoru gibi metrikler kullanılır. Fitness fonksiyonu, modelin başarısını değerlendiren ölçütleri tanımlar.
- Seçim: Fitness değerlerine göre, daha iyi olan bireyler seçilir. Bu, doğal seleksiyon ilkesine benzer. En iyi modeller, bir sonraki nesilde daha fazla yer alır.
- Çaprazlama (Crossover): Seçilen modellerin (bireylerin) parametreleri birbirleriyle "çaprazlanarak" yeni bireyler (model kombinasyonları) oluşturulur. Bu adımda, genetik algoritmaların çaprazlama süreçlerine benzer şekilde, yeni bir nesil model yapılandırılır.
- Mutasyon: Çaprazlama sonrası, bazı bireylerde rastgele değişiklikler (mutasyonlar) yapılır. Bu, yeni ve farklı çözüm adaylarını keşfetmek için gereklidir. Mutasyon, model parametrelerinde küçük değişiklikler yapar.
- Yeni Nesil ve Tekrar: Yeni nesil, önceki nesilden daha iyi çözümler sunduğu sürece evrimleşir. Bu süreç, belirli bir sayıda nesil geçene kadar veya belirli bir kriter sağlanana kadar devam eder.
- Sonuç: TPOT, en iyi performans gösteren modeli ve parametre kombinasyonunu sunar. Bu model, doğruluk, f1 skoru, recall gibi metriklere göre değerlendirilmiş olur.

![image](https://github.com/akay35/hiperparametre-optimizasyon-metodlari/blob/main/evolutionary%20algorithms.png)

##### TPOT Parametreleri
- generations: TPOT’in kaç nesil boyunca evrimleşeceğini belirler. Yüksek sayılar daha fazla hesaplama gücü gerektirir, ancak daha iyi sonuçlar elde etme olasılığı artırır.
- population_size: Her nesilde yer alacak model sayısını belirtir. Daha fazla model daha fazla keşif yapmanızı sağlar.
- random_state: Sonuçların tekrar edilebilirliğini sağlar.
- scoring: Hangi skor metriği ile modeli değerlendireceğini belirler (örneğin, doğruluk, f1 skoru vb.).

TPOT, çok sayıda model ve parametreyi test ettiği için yüksek hesaplama gücü gerektirir, büyük veri setleri veya çok sayıda nesil ile işlem yapıldığında, işlem süresi oldukça uzun olabilir.

---

### 6. Tree-structured Parzen Estimator (TPE)
###### Hiperparametreleri sistematik bir şekilde aramak yerine, olasılıksal bir model kullanarak en iyi hiperparametreleri tahmin eder. 
###### Bu yöntem, Bayesian optimization çerçevesinde çalışır ve hiperparametre arama işlemini daha etkili ve verimli hale getirir.
###### TPE, özellikle Hyperopt kütüphanesinde bir optimizasyon algoritması olarak yaygınca kullanılır. 

##### TPE Çalışma Prensibi
- Hiperparametre Alanını Tanımlama: Hiperparametreler için bir arama alanı (search space) belirlenir. Bu alan, sürekli (örneğin öğrenme oranı), ayrık (örneğin katman sayısı), veya kategorik (örneğin aktivasyon fonksiyonu) değerlerden oluşabilir.
- İlk Rastgele Denemeler: TPE, başlangıçta hiperparametre kombinasyonlarını rastgele seçer ve bu kombinasyonları değerlendirir (örneğin, çapraz doğrulama sonucu ile). Bu denemeler, olasılık dağılımlarını oluşturmak için veri sağlar.
- Performans Dağılımlarını Modelleme: Hedef performans (örneğin, doğruluk) için bir eşik değeri y* belirlenir. Bu değer, genellikle tüm hedef değerlerin belirli bir yüzdelik dilimidir (örneğin, ilk %20'lik dilim).
- Daha sonra, hiperparametreler iki gruba ayrılır:

𝑙 (𝑥): 𝑦>𝑦∗ (daha kötü performans)
𝑔(𝑥): 𝑦≤𝑦∗ (daha iyi performans)

- Yeni Hiperparametre Önerileri: TPE, daha önceki denemelerde yüksek performans göstermiş hiperparametre değerlerinin etrafında yoğunlaşan yeni hiperparametreler önerir. Bu, 𝑔(𝑥) / 𝑙(𝑥) oranını maksimize edecek şekilde yapılır.
- Deneme ve Güncelleme: Önerilen hiperparametreler denendikten sonra sonuçlar kaydedilir ve olasılık modelleri güncellenir. Bu işlem, belirlenen bir iterasyon veya zaman sınırına kadar devam eder.

##### Tree-structured Parzen Estimator (TPE) Parametre Ayarlamaları
###### 1. Hiperparametre Arama Alanı (Search Space)
- hp.choice(label, options)
Ayrık (discrete) değerler arasından seçim yapar. Örneğin, karar ağacının criterion parametresi ['gini', 'entropy'] olabilir.
###### 'criterion': hp.choice('criterion', ['gini', 'entropy'])

- hp.uniform(label, low, high)
Belirtilen alt ve üst sınır arasında sürekli bir aralıkta rastgele değer seçer. Örneğin, öğrenme oranı (learning_rate) genellikle bu şekilde tanımlanır.
###### 'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)

- hp.quniform(label, low, high, q)
Belirtilen sürekli aralıkta sabit artışlarla değer seçer. Örneğin, n_estimators için tam sayılar gereklidir, bu yüzden q ile adım büyüklüğü belirlenir.
###### 'n_estimators': hp.quniform('n_estimators', 10, 200, 10)

- hp.loguniform(label, low, high)
Logaritmik ölçekli sürekli bir aralıkta rastgele değer seçer. Öğrenme oranı gibi parametrelerde çok küçük değerler önemli olabilir.
###### 'learning_rate': hp.loguniform('learning_rate', -3, 0)  # 0.001 ile 1 arasında

- hp.randint(label, upper)
Belirtilen üst sınıra kadar tam sayı değerler seçer.
###### 'max_depth': hp.randint('max_depth', 20)


