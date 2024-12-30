import streamlit as st
import os
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests

llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class CompanyOverviewArgsSchema(BaseModel):
    symbol : str = Field(description="Stock symbol of the company. Example : AAPL, TSLA")

class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description= """
    Use this to get an overview of the financials of the company. 
    You should enter a stock symbol."""
    args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description= """
    Use this to get a income statement of the financials of the company. 
    You should enter a stock symbol."""
    args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()['annualReports']
    
class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description= """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol."""
    args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        res = r.json()
        return list(res['Weekly Time Series'].items())[:200]

class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = "Use this tool to find the stock market symbol for a company.\nIt takes a query as an argument.\nExample query: Stock Market Symbol for Apple Company"
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

agent = initialize_agent(
    llm=llm, 
    verbose=True,
    agent = AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
        
            You are a hedge fund manager.

            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                                            
            Consider the performance of a stock, the company overview and the income statement.
                                            
            Be assertive in your judgement and recommend the stock or advise the user against it.
                                            
            Also, the final answer should be Korean.
            """
        )
    },
)

st.set_page_config(
    page_icon="😀",
    page_title="InvestorGPT"
)

st.markdown(
    """
    # InvestorGPT

    Welcome to InvestorGPT.

    Write down the name of a company and our Agent will do the research for you.
    """
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)

    st.write(result['output'].replace("$", "\n$"))