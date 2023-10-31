#' @export
hf_api = function(inputs = NULL, url = NULL, filename = NULL, ...){
  require(httr);require(jsonlite)
  hf_opts = list()
  if (Sys.getenv("HF_API_TOKEN") != "") {
    config = add_headers(Authorization = paste("Bearer", Sys.getenv("HF_API_TOKEN")))
  } else {
    config = NULL
  }

  params = c(list(inputs = inputs), list(...))

  if(!is.null(filename)){
    body = read_file_raw(filename)
  } else {
    body = jsonlite::toJSON(c(params, hf_opts))
  }

  req = POST(url, body = body, config = config)
  response = fromJSON(content(req, "text"))
  if(is.list(response) && !is.null(response$estimated_time)){
    print(paste("Waiting", round(as.numeric(response$estimated_time)), "seconds for model to load."))
    Sys.sleep(response$estimated_time)
    hf_api(inputs, url, filename, ...)
  } else {
    response
  }
}

#' @export
text_embeddings = function(txt, url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L12-v2"){
  hf_api(inputs = txt, url = url) %>%
    as_tibble()
}

#' @export
text_ner = function(txt, url = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"){
  res_api <- hf_api(inputs = txt, url = url)

  d = res_api |>
    map2_dfr(txt, ~ .x %>% mutate(text = .y))

  return(d)
}

#' @export
text_classification = function(txt, url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"){
  res_api  <- hf_api(inputs = txt, url = url)

  d <- res_api |>
    map_dfr(~ .x |>
    pivot_wider(names_from = "label", values_from = "score")) |>
    bind_cols(text = txt)

  return(d)
}

#' @export
text_zeroshot = function(txt, labels, url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"){
  hf_api(inputs = txt, url = url, parameters = list(candidate_labels = labels)) %>%
    as_tibble()
}

#' @export
image_classification = function(filename, url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"){
  hf_api(filename = filename, url = url) %>%
    as_tibble()
}

#' @export
audio_classification = function(filename, url = "https://api-inference.huggingface.co/models/superb/hubert-large-superb-er"){
  hf_api(filename = filename, url = url) %>%
    as_tibble()
}
