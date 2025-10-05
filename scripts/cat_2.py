import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import yake
from typing import List, Dict, Tuple, Any
import warnings
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from catboost import CatBoostRegressor
import pickle
import re
import spacy

np.random.seed(42)

class MinimalTickerExtractor:
    def __init__(self, ticker_mapping: Dict[str, str]):
        self.ticker_mapping = ticker_mapping
        self.all_tickers = list(ticker_mapping.values())
        self.search_dict = {}
        for company, ticker in ticker_mapping.items():
            self.search_dict[company.lower()] = ticker
            variants = {
                'сбербанк': ['сбер', 'банк', 'финансы', 'кредит', 'ипотека', 'sberbank'],
                'газпром': ['газ', 'трубопровод', 'газодобыча', 'gazprom'],
                'лукойл': ['нефть', 'бензин', 'топливо', 'lukoil'],
                'норильский никель': ['никель', 'медь', 'металл', 'nornickel'],
                'роснефть': ['нефть', 'роснефти', 'rosneft'],
                'аэрофлот': ['авиа', 'рейс', 'аэропорт', 'aeroflot'],
                'московская биржа': ['биржа', 'мосбиржа', 'торги', 'moex'],
                'втб': ['втб банк', 'vtb'],
                'магнит': ['ритейл', 'магазин', 'сеть магазинов', 'magnit'],
                'мтс': ['связь', 'мобильная связь', 'mts'],
                'тинькофф': ['банк', 'онлайн банк', 'tinkoff'],
                'яндекс': ['поиск', 'интернет', 'yandex'],
            }
            if company in variants:
                for variant in variants[company]:
                    self.search_dict[variant] = ticker
        self.pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.search_dict.keys())) + r'|[A-Z]{3,5})\b',
            re.IGNORECASE
        )

    def find_tickers(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        found = set()
        for match in self.pattern.findall(text.lower()):
            if match in self.search_dict:
                found.add(self.search_dict[match])
            elif match.upper() in self.all_tickers:
                found.add(match.upper())
        return list(found)

def add_tickers_to_dataframe(
    df: pd.DataFrame,
    text_column: str,
    ticker_mapping: Dict[str, str]
) -> pd.DataFrame:
    extractor = MinimalTickerExtractor(ticker_mapping)
    df = df.copy()
    df['tickers'] = df[text_column].apply(extractor.find_tickers)
    return df

class EnhancedNewsAnalyzer:
    def __init__(self):
        pass
    
    def analyze_sentiment_improved(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._neutral_result()
        
        text_lower = text.lower()
        
        positive_indicators = {
            'выгодно': 4, 'выгодн': 3, 'преимуществ': 3, 'перспектив': 3,
            'рекомендац': 4, 'покупай': 4, 'инвестиру': 3, 'лидер': 3,
            'рекорд': 4, 'прорыв': 4, 'успешн': 3, 'эффективн': 2,
            'прибыль': 3, 'доход': 3, 'выручк': 3, 'дивиденд': 3,
            'рентабельност': 3, 'доходност': 3, 'профицит': 3,
            'стабильн': 2, 'комфортн': 2, 'уверен': 2,
            'рост': 3, 'увелич': 3, 'повыш': 3, 'улучш': 3,
            'развит': 2, 'расширен': 2, 'прогресс': 2,
            'позитив': 3, 'оптимизм': 3, 'перспективн': 3,
            'сильн': 2, 'успех': 3, 'достижен': 2
        }
        
        negative_indicators = {
            'проблем': 3, 'риск': 3, 'убыток': 4, 'потер': 3,
            'кризис': 4, 'опасн': 3, 'угроз': 3, 'сложност': 2,
            'трудност': 2, 'нестабильн': 3, 'волатильн': 2,
            'паден': 3, 'сниж': 3, 'сокращ': 3, 'уменьш': 3,
            'ухудш': 3, 'просадк': 3, 'обвал': 4,
            'отрицательн': 3, 'негативн': 3, 'плох': 2,
            'слаб': 2, 'критич': 3, 'неблагоприятн': 3
        }
        
        intensifiers = {
            'очень': 1.5, 'крайне': 2.0, 'сильно': 1.5, 'значительн': 1.5,
            'существенн': 1.5, 'резк': 1.5, 'катастрофич': 2.0, 'рекордн': 1.5,
            'масштабн': 1.3, 'высок': 1.2, 'больш': 1.1
        }
        
        context_phrases = {
            'positive': [
                'смотрятся выгодно', 'можно отметить', 'отдельно отметим',
                'комфортная долговая', 'высокая рентабельность', 'стабильный денежный поток',
                'высокие дивиденды', 'дивидендная доходность', 'превысить 10%'
            ],
            'negative': [
                'сложился неблагоприятно', 'сопряжено со сложностями', 
                'основные риски', 'ухудшением конъюнктуры'
            ]
        }
        
        positive_score = 0
        negative_score = 0
        
        for word, weight in positive_indicators.items():
            if word in text_lower:
                count = text_lower.count(word)
                for intensifier, multiplier in intensifiers.items():
                    if f"{intensifier} {word}" in text_lower:
                        positive_score += count * weight * multiplier
                        break
                else:
                    positive_score += count * weight
        
        for word, weight in negative_indicators.items():
            if word in text_lower:
                count = text_lower.count(word)
                for intensifier, multiplier in intensifiers.items():
                    if f"{intensifier} {word}" in text_lower:
                        negative_score += count * weight * multiplier
                        break
                else:
                    negative_score += count * weight
        
        for phrase in context_phrases['positive']:
            if phrase in text_lower:
                positive_score += 5
        
        for phrase in context_phrases['negative']:
            if phrase in text_lower:
                negative_score += 5
        
        total_score = positive_score - negative_score
        
        if total_score >= 2:
            sentiment = 'positive'
            confidence = min((total_score + 5) / 15, 0.95)
        elif total_score <= -3:
            sentiment = 'negative'
            confidence = min((abs(total_score) + 5) / 15, 0.95)
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        emotional_score = total_score / max(positive_score + negative_score + 1, 10)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'emotional_score': emotional_score,
            'method': 'improved_financial'
        }
    
    def has_financial_context_enhanced(self, text: str) -> bool:
        if not text:
            return False
        
        text_lower = text.lower()
        
        financial_terms = [
            'акци', 'акция', 'акций', 'дивиденд', 'котировк', 'курс', 'цена',
            'прибыль', 'убыток', 'выручк', 'доход', 'отчетност', 'квартал',
            'млрд', 'млн', 'долл', 'рубл', 'евро', 'процент', 'пункт',
            'рост', 'падени', 'инвест', 'портфель', 'рынок', 'бирж',
            'сделка', 'покупк', 'продаж', 'волатильн', 'ликвидност',
            'финанс', 'бюджет', 'капзатрат', 'актив', 'пассив', 'баланс',
            'выплат', 'затрат', 'издержк', 'рентабельност', 'доходност',
            'долгов', 'нагрузк', 'денежн', 'поток', 'fcf', 'ebitda',
            'мультипликатор', 'конъюнктур', 'налогообложен', 'эмитент',
            'эмисси', 'капитализация', 'облигац', 'купон',
            'кредит', 'заем', 'депозит', 'вклад', 'ипотек', 'рефинанс',
            'ввп', 'инфляц', 'ключев', 'ставк', 'цб', 'центробанк',
            'нефт', 'газ', 'энерг', 'метал', 'горнодобыва', 'телеком',
            'ритейл', 'строительств', 'транспорт', 'хими'
        ]
        
        financial_phrases = [
            'денежный поток', 'долговая нагрузка', 'чистая прибыль',
            'валовая выручка', 'операционная деятельность', 'финансовый результат',
            'отчетность по мсфо', 'дивидендная политика', 'рыночная капитализация',
            'котировки акций', 'биржевые торги', 'инвестиционный портфель',
            'финансовый анализ', 'экономический показатель', 'макроэкономическая ситуация'
        ]
        
        financial_terms_count = 0
        
        for term in financial_terms:
            if term in text_lower:
                financial_terms_count += 1
        
        for phrase in financial_phrases:
            if phrase in text_lower:
                financial_terms_count += 3
        
        financial_contexts = [
            r'\b(?:отчет|отчетность|баланс|прибыль|убыток)[^.]{0,100}',
            r'\b(?:цена|курс|котировки)[^.]{0,100}(?:акци|акций|облигац)',
            r'\b(?:дивиденд|выплата)[^.]{0,100}(?:акционер|прибыль)',
        ]
        
        for pattern in financial_contexts:
            if re.search(pattern, text_lower):
                financial_terms_count += 2
        
        return financial_terms_count >= 3
    
    def _neutral_result(self):
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'emotional_score': 0.0,
            'method': 'fallback'
        }
    
    def analyze_news_comprehensive(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'emotional_score': 0.0,
                'has_financial_context': False,
                'method': 'fallback'
            }
        
        sentiment_result = self.analyze_sentiment_improved(text)
        has_financial_context = self.has_financial_context_enhanced(text)
        
        return {
            **sentiment_result,
            'has_financial_context': has_financial_context
        }

def reanalyze_with_enhancements(df) -> pd.DataFrame:
    analyzer = EnhancedNewsAnalyzer()
        
    analysis_results = df['publication'].apply(analyzer.analyze_news_comprehensive)
    
    df['sentiment'] = analysis_results.apply(lambda x: x['sentiment'])
    df['sentiment_confidence'] = analysis_results.apply(lambda x: x['confidence'])
    df['emotional_score'] = analysis_results.apply(lambda x: x['emotional_score'])
    df['analysis_method'] = analysis_results.apply(lambda x: x['method'])
    df['has_financial_context'] = analysis_results.apply(lambda x: x['has_financial_context'])
    
    return df

def build_ticker_aliases(tickers):
    base = {
        'AFLT': ['аэрофлот', 'aeroflot'],
        'ALRS': ['алроса', 'alrosa'],
        'CHMF': ['северсталь', 'severstal'],
        'GAZP': ['газпром', 'gazprom'],
        'GMKN': ['норникель', 'норильский никель', 'nornickel', 'norilsk nickel'],
        'LKOH': ['лукойл', 'lukoil'],
        'MAGN': ['ммк', 'магнитогорский металлургический комбинат', 'mmk'],
        'MGNT': ['магнит', 'magnit'],
        'MOEX': ['мосбиржа', 'московская биржа', 'moex'],
        'MTSS': ['мтс', 'mts'],
        'NVTK': ['новатэк', 'novatek'],
        'PHOR': ['фосагро', 'phosagro'],
        'PLZL': ['полюс', 'polyus'],
        'ROSN': ['роснефть', 'rosneft'],
        'RUAL': ['русал', 'rusal'],
        'SBER': ['сбер', 'сбербанк', 'sber', 'sberbank'],
        'SIBN': ['газпром нефть', 'gazprom neft'],
        'T':    ['тинькофф', 'т-банк', 'tinkoff', 't-bank'],
        'VTBR': ['втб', 'vtb'],
    }
    return {t: list(set(base.get(t, []) + [t.lower()])) for t in tickers}

def preprocess_sentiment_dummies(news_df):
    df = news_df.copy()
    
    if 'sentiment' in df.columns:
        sentiment_dummies = pd.get_dummies(df['sentiment'], prefix='sentiment')
        df = pd.concat([df, sentiment_dummies], axis=1)
    
    return df

def tag_news_with_tickers(news_df, ticker_aliases, text_cols=('title','publication')):
    news = news_df.copy()
    news['publish_date'] = pd.to_datetime(news['publish_date'])
    news = preprocess_sentiment_dummies(news)
    
    mentioned_cols = [col for col in news.columns if col.startswith('mentioned_')]
    
    if mentioned_cols:
        def get_mentioned_tickers(row):
            tickers_found = []
            for col in mentioned_cols:
                if row[col] == 1:
                    ticker = col.replace('mentioned_', '')
                    tickers_found.append(ticker)
            return tickers_found
        
        news['tickers'] = news.apply(get_mentioned_tickers, axis=1)
    else:
        news['_text'] = ''
        for c in text_cols:
            if c in news.columns:
                news['_text'] = (news['_text'] + ' ' + news[c].astype(str)).str.lower()
        
        def find_tickers(text):
            if not isinstance(text, str) or not text:
                return []
            found = []
            for tkr, keys in ticker_aliases.items():
                if any(k and k in text for k in keys):
                    found.append(tkr)
            return list(set(found))
        
        news['tickers'] = news['_text'].apply(find_tickers)
        news = news.drop(columns=['_text'])
    
    return news

def aggregate_news_features_by_ticker(news_tagged):
    df = news_tagged.copy()
    df['date'] = df['publish_date'].dt.normalize()

    exploded = df.explode('tickers')
    exploded = exploded[exploded['tickers'].notna() & (exploded['tickers'] != '')]

    sentiment_cols = [col for col in exploded.columns if col.startswith('sentiment_')]

    aggregation_dict = {
        'title': 'count',
    }
    if 'sentiment_confidence' in exploded.columns:
        aggregation_dict['sentiment_confidence'] = 'mean'
    if 'emotional_score' in exploded.columns:
        aggregation_dict['emotional_score'] = 'mean'
    if 'has_financial_context' in exploded.columns:
        aggregation_dict['has_financial_context'] = 'sum'

    for col in sentiment_cols:
        aggregation_dict[col] = 'sum'

    agg = exploded.groupby(['tickers', 'date']).agg(aggregation_dict).reset_index()

    rename_map = {
        'tickers': 'ticker',
        'title': 'news_count',
        'sentiment_confidence': 'avg_sentiment_confidence',
        'emotional_score': 'avg_emotional_score',
        'has_financial_context': 'financial_news_count'
    }
    agg = agg.rename(columns=rename_map)

    if 'financial_news_count' in agg.columns:
        agg['financial_news_ratio'] = agg['financial_news_count'] / agg['news_count']
    else:
        agg['financial_news_count'] = 0.0
        agg['financial_news_ratio'] = 0.0

    for col in sentiment_cols:
        if col in agg.columns:
            ratio_col = f'{col}_ratio'
            agg[ratio_col] = agg[col] / agg['news_count']

    agg = agg.fillna(0.0)

    return agg

def aggregate_news_counts_by_ticker(news_tagged):
    df = news_tagged.copy()
    df['date'] = df['publish_date'].dt.normalize()
    exploded = df.explode('tickers')
    exploded = exploded[exploded['tickers'].notna() & (exploded['tickers']!='')]
    agg = exploded.groupby(['tickers','date']).size().reset_index(name='news_count')
    agg = agg.rename(columns={'tickers':'ticker'})
    return agg

def add_news_features_by_ticker(candles_df, news_df, ticker_aliases):
    candles = candles_df.copy()
    candles['begin'] = pd.to_datetime(candles['begin'])
    candles['date'] = candles['begin'].dt.normalize()
    
    if news_df is None or len(news_df)==0:
        candles['news_count'] = 0.0
        candles['avg_sentiment_confidence'] = 0.0
        candles['avg_emotional_score'] = 0.0
        candles['financial_news_count'] = 0.0
        candles['financial_news_ratio'] = 0.0
        return candles.drop(columns=['date'])
    
    tagged = tag_news_with_tickers(news_df, ticker_aliases)
    news_features = aggregate_news_features_by_ticker(tagged)
    news_features['date'] += pd.Timedelta(days=1)
    
    out = candles.merge(news_features, on=['ticker','date'], how='left')
    
    news_cols = [col for col in news_features.columns if col not in ['ticker', 'date']]
    
    for col in news_cols:
        out[col] = out[col].fillna(0.0)
    
    expected_sentiments = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_mixed']
    for sent in expected_sentiments:
        if sent not in out.columns:
            out[sent] = 0.0
        if f'{sent}_ratio' not in out.columns:
            out[f'{sent}_ratio'] = 0.0
    
    return out.drop(columns=['date'])

def add_news_count_by_ticker(candles_df, news_df, ticker_aliases):
    candles = candles_df.copy()
    candles['begin'] = pd.to_datetime(candles['begin'])
    candles['date'] = candles['begin'].dt.normalize()
    if news_df is None or len(news_df)==0:
        candles['news_count'] = 0.0
        return candles.drop(columns=['date'])
    tagged = tag_news_with_tickers(news_df, ticker_aliases)
    per_ticker = aggregate_news_counts_by_ticker(tagged)
    out = candles.merge(per_ticker, on=['ticker','date'], how='left')
    out['news_count'] = out['news_count'].fillna(0.0)
    return out.drop(columns=['date'])

FEATS = [
    'momentum_5', 'volatility_5', 'price_range',
    'news_count', 'financial_news_count', 'financial_news_ratio',
    'avg_sentiment_confidence', 'avg_emotional_score',
    'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_mixed',
    'sentiment_positive_ratio', 'sentiment_negative_ratio', 'sentiment_neutral_ratio', 'sentiment_mixed_ratio'
]

def create_features(df):
    df = df.copy()
    df['begin'] = pd.to_datetime(df['begin'])
    df = df.sort_values(['ticker','begin']).reset_index(drop=True)
    df['momentum_5'] = df.groupby('ticker')['close'].pct_change(5).shift(1)
    ret1 = df.groupby('ticker')['close'].pct_change()
    df['volatility_5'] = ret1.groupby(df['ticker']).rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
    df['price_range'] = (df['high'] - df['low'])/df['close']
    
    for c in ['momentum_5','volatility_5','price_range']:
        df[c] = df[c].fillna(0.0)
    
    return df

def create_targets(df, horizons=(1,20)):
    out = df.copy()
    for h in horizons:
        out[f'target_return_{h}d'] = out.groupby('ticker')['close'].pct_change(h).shift(-h)
    return out

def fit(candles_train, news_train=None, split_date='2024-09-08', model_path='model.pkl'):
    df = candles_train.copy()
    df['begin'] = pd.to_datetime(df['begin'])
    cutoff = pd.to_datetime(split_date)
    df = df[df['begin'] <= cutoff].copy()

    if len(df) == 0:
        raise ValueError("Train slice after split_date is empty; adjust split_date or inputs.")

    aliases = build_ticker_aliases(sorted(df['ticker'].unique()))

    df = create_features(df)
    df = add_news_features_by_ticker(df, news_train, aliases)
    df = create_targets(df, horizons=(1, 20))

    mask = ~df[[f'target_return_{h}d' for h in (1, 20)]].isna().any(axis=1)
    dft = df.loc[mask].reset_index(drop=True)

    if len(dft) == 0:
        raise ValueError("No valid rows with both targets h=1 and h=20 in train.")

    X = dft.drop(columns=['begin', 'ticker'] + [f'target_return_{h}d' for h in (1, 20)]).values
    feature_names = [c for c in dft.columns if c not in ['begin', 'ticker',
                                                        'target_return_1d', 'target_return_20d']]

    models = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for h in (1, 20):
        y = dft[f'target_return_{h}d'].values

        reg = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )

        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            reg.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
            preds = reg.predict(Xval)
            fold_rmse = np.sqrt(np.mean((preds - yval) ** 2))
            scores.append(fold_rmse)

        models[f'reg_{h}'] = reg

    with open(model_path, 'wb') as f:
        pickle.dump({
            'features': feature_names,
            'models': models,
            'aliases': aliases,
            'split_date': str(split_date)
        }, f)

def _safe_write_csv(df, path):
    df.to_csv(path, index=False, sep=',', encoding='utf-8-sig', lineterminator='\n')

def predict_on_cutoff(candles_test, news_test=None, model_path='model.pkl', output_path='submission.csv'):
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)
        
    feats   = payload['features']
    models  = payload['models']
    aliases = payload['aliases']
    split_date = pd.to_datetime(payload['split_date'])
    
    df = candles_test.copy()
    df['begin'] = pd.to_datetime(df['begin'])
    df = df[df['begin'] > split_date].copy()
    
    df = create_features(df)
    df = add_news_features_by_ticker(df, news_test, aliases)
    
    X = df[feats].values
    
    for h in (1, 20):
        reg = models[f'reg_{h}']
        df[f'pred_return_{h}d'] = reg.predict(X)
    
    out = df[['ticker', 'begin', 'pred_return_1d', 'pred_return_20d']]
    out.to_csv(output_path, index=False)
    return out

def predict_for_date(candles_df, news_df, model_path, prediction_date, output_path='submission.csv'):
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)
        
    feats = payload['features']
    models = payload['models']
    aliases = payload['aliases']
    
    prediction_date = pd.to_datetime(prediction_date)
    
    df = candles_df.copy()
    df['begin'] = pd.to_datetime(df['begin'])
    
    df = df[df['begin'] == prediction_date].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data available for prediction date {prediction_date}")
    
    df = create_features(df)
    df = add_news_features_by_ticker(df, news_df, aliases)
    
    missing_features = set(feats) - set(df.columns)
    if missing_features:
        for feature in missing_features:
            df[feature] = 0.0
    
    X = df[feats].values
    tickers = df['ticker'].values
    
    predictions = {}
    for h in range(1, 21):
        model_key = f'reg_{h}'
        if model_key in models:
            predictions[f'p{h}'] = models[model_key].predict(X)
        else:
            available_horizons = [int(k.split('_')[1]) for k in models.keys() if k.startswith('reg_')]
            closest_h = min(available_horizons, key=lambda x: abs(x - h))
            predictions[f'p{h}'] = models[f'reg_{closest_h}'].predict(X)
    
    result_df = pd.DataFrame({'ticker': tickers})
    for h in range(1, 21):
        result_df[f'p{h}'] = predictions[f'p{h}']
    
    result_df.to_csv(output_path, index=False)
    
    return result_df

class NewsEnhancedSolution:
    def __init__(self, model_path: str = 'model.pkl'):
        self.model_path = model_path

    def load_data(self, train_candles_path: str,
                  public_test_path: str,
                  private_test_path: str,
                  train_news_path: str = None,
                  test_news_path: str = None):
        
        self.train_candles = pd.read_csv(train_candles_path)
        self.train_candles['begin'] = pd.to_datetime(self.train_candles['begin'])

        public_test_df = pd.read_csv(public_test_path)
        public_test_df['begin'] = pd.to_datetime(public_test_df['begin'])

        private_test_df = pd.read_csv(private_test_path)
        private_test_df['begin'] = pd.to_datetime(private_test_df['begin'])

        self.test_candles = pd.concat([public_test_df, private_test_df], ignore_index=True)
        
        if train_news_path:
            self.train_news = pd.read_csv(train_news_path)
        else:
            self.train_news = None
            
        if test_news_path:
            self.test_news = pd.read_csv(test_news_path)
        else:
            self.test_news = None

        self.full_candles = pd.concat([self.train_candles, self.test_candles], ignore_index=True)
        self.full_candles = self.full_candles.sort_values(['ticker', 'begin'])

    def prepare_news_features(self, news_df):
        if news_df is None or len(news_df) == 0:
            return None
            
        ticker_mapping = {
            'аэрофлот': 'AFLT',
            'алроса': 'ALRS', 
            'газпром': 'GAZP',
            'лукойл': 'LKOH',
            'сбербанк': 'SBER',
            'роснефть': 'ROSN',
            'норильский никель': 'GMKN',
            'московская биржа': 'MOEX',
            'втб': 'VTBR',
            'магнит': 'MGNT',
            'мтс': 'MTSS',
            'татнефть': 'TATN',
            'фосагро': 'PHOR',
            'полюс': 'PLZL',
            'русал': 'RUAL',
            'северсталь': 'CHMF',
            'новатэк': 'NVTK'
        }
        
        news_with_tickers = add_tickers_to_dataframe(
            df=news_df,
            text_column='publication',
            ticker_mapping=ticker_mapping
        )
        
        news_features = reanalyze_with_enhancements(news_with_tickers)
        return news_features

    def train_model(self):
        train_news_features = self.prepare_news_features(self.train_news)
        
        fit(
            candles_train=self.train_candles,
            news_train=train_news_features,
            split_date='2024-09-08',
            model_path=self.model_path
        )

    def predict(self):
        test_news_features = self.prepare_news_features(self.test_news)
        
        prediction_dates = self.test_candles['begin'].unique()
        
        all_predictions = []
        
        for date in prediction_dates:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            
            predictions = predict_for_date(
                candles_df=self.test_candles,
                news_df=test_news_features,
                model_path=self.model_path,
                prediction_date=date_str,
                output_path=None
            )
            
            predictions['begin'] = date
            all_predictions.append(predictions)
        
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        return final_predictions

    def save_submission(self, predictions, output_path: str = "submission.csv"):
        submission = predictions.copy()
        submission.to_csv(output_path, index=False)

    def run(self, train_candles_path: str, public_test_path: str,
            private_test_path: str, output_path: str = "submission.csv",
            train_news_path: str = None, test_news_path: str = None):
        
        self.load_data(
            train_candles_path=train_candles_path,
            public_test_path=public_test_path,
            private_test_path=private_test_path,
            train_news_path=train_news_path,
            test_news_path=test_news_path
        )
        
        self.train_model()
        
        predictions = self.predict()
        
        self.save_submission(predictions, output_path)
        
        return predictions

if __name__ == "__main__":
    solution = NewsEnhancedSolution(model_path='model.pkl')
    
    candles = pd.read_csv('../data/new/forecast_data/candles.csv')
    candles_2 = pd.read_csv('../data/new/forecast_data/candles_2.csv')
    news = pd.read_csv('../data/new/forecast_data/news.csv')
    news_2 = pd.read_csv('../data/new/forecast_data/news_2.csv')
    
    ticker_mapping = {
        'аэрофлот': 'AFLT',
        'алроса': 'ALRS', 
        'газпром': 'GAZP',
        'лукойл': 'LKOH',
        'сбербанк': 'SBER',
        'роснефть': 'ROSN',
        'норильский никель': 'GMKN',
        'московская биржа': 'MOEX',
        'втб': 'VTBR',
        'магнит': 'MGNT',
        'мтс': 'MTSS',
        'татнефть': 'TATN',
        'фосагро': 'PHOR',
        'полюс': 'PLZL',
        'русал': 'RUAL',
        'северсталь': 'CHMF',
        'новатэк': 'NVTK'
    }

    news_train_with_flags = add_tickers_to_dataframe(
        df=news,
        text_column='publication',
        ticker_mapping=ticker_mapping
    )
    
    news_features = reanalyze_with_enhancements(news_train_with_flags)
    
    fit(candles, news_features, split_date='2024-09-08', model_path='model.pkl')
    
    results = predict_for_date(
        candles_df=candles,
        news_df=news_features, 
        model_path='model.pkl',
        prediction_date='2024-09-09',
        output_path='submission.csv'
    )