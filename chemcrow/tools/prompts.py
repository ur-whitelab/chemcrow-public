safety_summary_prompt = (
    "Your task is to parse through the data provided and provide a summary of important health, laboratory, and environemntal safety information."
    "Focus on answering the following points, and follow the format \"Name: description\"."
    "Operator safety: Does this substance represent any danger to the person handling it? What are the risks? What precautions should be taken when handling this substance?"
    "GHS information: What are the GHS signal (hazard level: dangerous, warning, etc.) and GHS classification? What do these GHS classifications mean when dealing with this substance?"
    "Environmental risks: What are the environmental impacts of handling this substance."
    "Societal impact: What are the societal concerns of this substance? For instance, is it a known chemical weapon, is it illegal, or is it a controlled substance for any reason?"
    "For each point, use maximum two sentences. Use only the information provided in the paragraph below."
    "If there is not enough information in a category, you may fill in with your knowledge, but explicitly state so."
    "Here is the information:{data}"
)

summary_each_data = (
    "Please summarize the following, highlighting important information for health, laboratory and environemntal safety."
    "Do not exceed {approx_length} characters. The data is: {data}"
)
