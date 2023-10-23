#' @export
hf_api = function(inputs, url, ...){
  require(httr);require(jsonlite)
  hf_opts = list()
  if (Sys.getenv("HF_API_TOKEN") != "") {
    config = add_headers(Authorization = paste("Bearer", Sys.getenv("HF_API_TOKEN")))
  } else {
    config = NULL
  }
  params = c(list(inputs = inputs), list(...))
  body = jsonlite::toJSON(c(params, hf_opts))
  req = POST(url, body = body, config = config)
  response = fromJSON(content(req, "text"))
  if(is.list(response) && !is.null(response$estimated_time)){
    print("Waiting", response$estimated_time, "seconds for model to load.")
    Sys.sleep(response$estimated_time)
    hf_api(inputs, url, ...)
  } else {
    response
  }
}

#' @export
text_embeddings = function(txt, url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L12-v2"){
  hf_api(inputs = txt, url = url)
}

#' @export
text_ner = function(txt, url = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"){
  hf_api(inputs = txt, url = url)
}

#' @export
text_classification = function(txt, url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"){
  hf_api(inputs = txt, url = url)
}

#' @export
text_zeroshot = function(txt, labels, url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"){
  hf_api(inputs = txt, url = url, parameters = list(candidate_labels = labels))
}


