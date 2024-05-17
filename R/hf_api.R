#' @export
hf_api = function(inputs = NULL, url = NULL, filename = NULL, request_type = "default", ...){
  require(httr)
  require(jsonlite)

  hf_opts = list()
  if (Sys.getenv("HF_API_TOKEN") != "") {
    config = add_headers(Authorization = paste("Bearer", Sys.getenv("HF_API_TOKEN")))
  } else {
    config = NULL
  }

  # Adjust the headers based on request type
  if (request_type == "text_generation") {
    config <- add_headers(
      Authorization = paste("Bearer", Sys.getenv("HF_API_TOKEN")),
      `Content-Type` = "application/json"
    )
  }

  params = c(list(inputs = inputs), list(...))

  # Convert inputs to string if not already
  if (!is.character(inputs)) {
    inputs <- as.character(inputs)
  }

  # Construct the body based on request type
  if (request_type == "text_generation") {
    body <- toJSON(list(
      inputs = inputs,
      parameters = list(
        return_full_text = FALSE
      )
    ), auto_unbox = TRUE, pretty = TRUE)
  } else if (!is.null(filename) && is.character(params$parameters$candidate_labels)) {
    # Handle image upload with candidate_labels
    image_data <- base64enc::base64encode(filename)
    input_list <- list(
      parameters = list(candidate_labels = params$parameters$candidate_labels),
      image = image_data
    )
    body <- jsonlite::toJSON(input_list, auto_unbox = TRUE)
  } else if (!is.null(filename)) {
    body = read_file_raw(filename)
  } else {
    body <- jsonlite::toJSON(c(params, hf_opts), auto_unbox = TRUE)
  }


  #print(paste("Request body:", body)) # Debug print

  req = POST(url, body = body, config = config, encode = "json")

  response_text = content(req, "text", encoding = "UTF-8")

  print(paste("Response text:", response_text)) # Debug print

  # Try to parse JSON response
  response <- tryCatch({
    fromJSON(response_text)
  }, error = function(e) {
    stop(paste("Failed to parse JSON response:", response_text))
  })

  if(is.list(response) && !is.null(response$estimated_time)){
    print(paste("Waiting", round(as.numeric(response$estimated_time)), "seconds for model to load."))
    Sys.sleep(response$estimated_time)
    hf_api(inputs, url, filename, request_type, ...)
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
  res_api <- hf_api(inputs = txt, url = url) %>%
    as_tibble() %>%
    mutate(text = txt)

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
    as_tibble() %>%
    unnest(cols = c("labels", "scores"))
}

#' @export
text_generation = function(txt, url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct", request_type = "text_generation"){
  instruction = paste0("<|user|> You are a helpful chatbot assistant which provides answer based on the context given. Do not give any extra information. Do not give the context again in your response.\n",
                      "Generate a concise and informative answer in less than 100 words for the given question.\n",
                      txt, "<|end|><|assistant|>")

  hf_api(inputs = instruction, url = url, request_type = request_type) %>%
    unlist() %>%
    as_tibble()
}


#' @export
image_zeroshot = function(filename, labels, url = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16"){
  hf_api(filename = filename, url = url, parameters = list(candidate_labels = labels)) %>% as_tibble()
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

#' @export
image_to_text = function(filename, url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"){
  hf_api(filename = filename, url = url) %>%
    as_tibble()
}
