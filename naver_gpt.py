import streamlit as st
import pandas as pd
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import re
import time
import random
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_company_data():
    df = pd.read_csv('company_codes.csv', dtype=str)
    return df

# 검색 옵션 생성
def create_search_options(df):
    options = [f"{row['회사명']} ({row['종목코드']})" for _, row in df.iterrows()]
    return options

# 선택된 옵션에서 종목코드 추출
def extract_code_from_option(option):
    return option.split('(')[-1].replace(')', '')

# 뉴스 크롤러 클래스
class NewsCrawler:
    def __init__(self, company_data):
        self.company_code_table = company_data
    
    def crawler(self, company_code, num_article):
        session = HTMLSession()
        done_page_num = 0
        num_per_page = 20
        num_page, remainder = divmod(num_article, num_per_page)
        num_page += 1
        article_result = []
        
        for page in range(done_page_num + 1, done_page_num + num_page + 1):
            try:
                url = f'https://finance.naver.com/item/news_news.nhn?code={company_code}&page={page}'
                source_code = session.get(url).text
                html = BeautifulSoup(source_code, "lxml")
                
                links = html.select('.title')
                link_result = []
                if page == num_page:
                    links = links[:remainder]
                for link in links:
                    add = 'https://finance.naver.com' + link.find('a')['href']
                    link_result.append(add)
            except Exception:
                pass
            
            for article_url in link_result:
                try:
                    article_source_code = session.get(article_url).html
                    redirect_url = article_source_code.search("top.location.href='{}';")[0]
                    article_source_code = session.get(redirect_url).text
                    article_html = BeautifulSoup(article_source_code, "lxml")
                    article_time = article_html.select('.media_end_head_info_datestamp_time')[0].get_text()
                    article_title = article_html.select('.media_end_head_title')[0].get_text().strip()
                    article_contents = article_html.select('._article_content')[0].get_text().strip()
                    article_contents = re.sub('\n', '', article_contents)
                    article_contents = re.sub('\t', '', article_contents)
                    
                    if "ⓒ" in article_contents:
                        article_contents = article_contents[:article_contents.index("ⓒ")]
                    
                    if len(article_contents) >= 1500:
                        article_contents = article_contents[:1500]

                    article_result.append([article_title, article_contents, article_time])
                    time.sleep(random.uniform(0.1, 0.7))
                except Exception:
                    pass

        return article_result
    
    def convert_company_to_code(self, company):
        if company in self.company_code_table['종목코드'].values:
            return company
        elif company in self.company_code_table['회사명'].values:
            return self.company_code_table[self.company_code_table['회사명'] == company]['종목코드'].values[0]
        else:
            return None
    
    def crawl_news(self, company, max_num=5):
        company_code = self.convert_company_to_code(company)

        if company_code:
            result = self.crawler(company_code, max_num)
            for i in range(len(result)):
                result[i].append(company)
            return result
        else:
            return []

# 뉴스 분석 함수
def analyze_news(news_data):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    template = """
    다음은 {company}에 관한 뉴스 기사입니다:

    {news_content}

    이 뉴스를 요약하고, 전반적인 분위기가 긍정적인지 부정적인지 분석해주세요. 
    또한 해당 분위기의 강도를 나타내세요. (숫자 1~10)
    강도는 해당 뉴스의 분위기가 주식 시장에 어느 정도의 강도로 적용될지를 뜻합니다.
    결과는 다음 형식으로 제시해주세요:

    요약: [뉴스 내용 요약]
    분위기: [긍정/부정/중립] [강도] 
    (이유 설명)

    한국어로 답변해주세요.
    """

    prompt = PromptTemplate(
        input_variables=["company", "news_content"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    full_content = "\n\n".join([f"제목: {news[0]}\n내용: {news[1]}" for news in news_data])
    company = news_data[0][3]

    with get_openai_callback() as cb:
        result = chain.run(company=company, news_content=full_content)
    
    return result, cb

# Streamlit 앱
def main():
    st.set_page_config(page_title="주식 뉴스 분석기", page_icon="📈", layout="wide")
    st.title("📊 주식 뉴스 분석기")

    # CSS를 사용하여 UI 개선
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #222;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 종목 데이터 로드
    company_data = load_company_data()
    search_options = create_search_options(company_data)

    # 사이드바에 입력 폼 배치
    with st.sidebar:
        st.header("📝 분석 설정")
        
        selected_option = st.selectbox(
            "종목명 또는 종목 코드를 선택하세요",
            options=search_options
        )
        company_code = extract_code_from_option(selected_option) if selected_option else None
        
        max_news = st.slider("분석할 뉴스 개수", 1, 20, 5)
        analyze_button = st.button("뉴스 분석하기")

    if analyze_button and company_code:
        with st.spinner("뉴스를 크롤링하고 분석 중입니다..."):
            # 뉴스 크롤링
            crawler = NewsCrawler(company_data)
            news_data = crawler.crawl_news(company_code, max_news)

            if news_data:
                # 뉴스 분석
                analysis_result, callback = analyze_news(news_data)

                # 결과 표시
                st.markdown("<p class='big-font'>분석 결과</p>", unsafe_allow_html=True)
                st.markdown("<div class='result-box'>" + analysis_result.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

                st.markdown("<p class='big-font'>사용 정보</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='result-box'>
                총 사용된 토큰 수: {callback.total_tokens}<br>
                프롬프트에 사용된 토큰 수: {callback.prompt_tokens}<br>
                답변에 사용된 토큰 수: {callback.completion_tokens}<br>
                호출에 청구된 금액(USD): ${callback.total_cost:.6f}<br>
                호출에 청구된 금액(KRW): ₩{callback.total_cost * 1300:.2f} (1 USD = 1300 KRW 가정)
                </div>
                """, unsafe_allow_html=True)

                # 개별 뉴스 표시
                st.markdown("<p class='big-font'>크롤링된 뉴스</p>", unsafe_allow_html=True)
                for news in news_data:
                    with st.expander(f"{news[0]} - {news[2]}"):
                        st.write(news[1])
            else:
                st.error("뉴스를 찾을 수 없습니다. 종목명 또는 종목 코드를 확인해주세요.")

if __name__ == "__main__":
    main()