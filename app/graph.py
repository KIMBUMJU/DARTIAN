from langgraph.graph import START, StateGraph
from app.state import State
from app.retriever import retrieve
from app.generator import generate
from IPython.display import Image, display

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# Mermaid 그래프 이미지를 파일로 저장
image_data = graph.get_graph().draw_mermaid_png()
output_path = "graph_image.png"