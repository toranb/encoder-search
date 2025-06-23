defmodule ExampleWeb.PageLive do
  use ExampleWeb, :live_view

  @max_length 100

  alias Example.Repo

  @impl true
  def mount(_, _, socket) do
    messages = []
    books = Example.Book |> Repo.all() |> Repo.preload(:verses)

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    model_data = File.read!("#{Path.dirname(__ENV__.file)}/bible_encoder")
    deserialized_data = Nx.deserialize(model_data)
    params = deserialized_data.model

    serving =
      Example.Generation.get_embeddings(params, tokenizer,
        compile: [batch_size: 512, sequence_length: [@max_length]],
        defn_options: [compiler: EXLA]
      )

    socket = socket |> assign(serving: serving, path: nil, lookup: nil, books: books, messages: messages, text: nil, loading: false, selected: nil, focused: false)

    {:ok, socket}
  end

  @impl true
  def handle_event("select_book", %{"id" => book_id}, socket) do
    book = socket.assigns.books |> Enum.find(&(&1.id == String.to_integer(book_id)))
    socket = socket |> assign(selected: book, result: nil)

    {:noreply, socket}
  end

  @impl true
  def handle_event("change_text", %{"message" => text}, socket) do
    socket = socket |> assign(text: text)
    {:noreply, socket}
  end

  @impl true
  def handle_event("add_message", _, %{assigns: %{loadingpdf: true}} = socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_event("add_message", _, %{assigns: %{loading: true}} = socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_event("add_message", %{"message" => ""}, socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_event("add_message", %{"message" => search}, socket) do
    messages = socket.assigns.messages
    selected = socket.assigns.selected
    serving = socket.assigns.serving

    message_id = Ecto.UUID.generate()
    new_messages = messages ++ [%{id: message_id, user_id: 1, text: search, inserted_at: DateTime.utc_now(), book_id: selected.id}]

    lookup =
      Task.async(fn ->
        encoder_result = Nx.Serving.run(serving, search)
        bm25_results = Example.Verse.search_keywords(selected.id, search)
        verse_ids = bm25_results |> Enum.map(fn {_, {verse_id, _, _, _, _}} -> verse_id end)
        colbert_results = Example.VerseToken.search(verse_ids, encoder_result.embedding)
        results = Example.Rank.rank_results(bm25_results, colbert_results, :weighted_sum, 0.7)

        {search, results}
      end)

    {:noreply, assign(socket, lookup: lookup, loading: true, text: nil, messages: new_messages)}
  end

  @impl true
  def handle_info({ref, {_search, results}}, socket) when socket.assigns.lookup.ref == ref do
    messages = socket.assigns.messages
    selected = socket.assigns.selected

    [{_, {_, chapter, verse, text, book_id}}] = results |> Enum.take(1)
    book_name = Example.Utils.book_name(book_id)
    result = "#{book_name} #{chapter}:#{verse}\n#{text}"
    message_id = Ecto.UUID.generate()
    new_messages = messages ++ [%{id: message_id, user_id: 2, text: result, inserted_at: DateTime.utc_now(), book_id: selected.id}]

    {:noreply, assign(socket, lookup: nil, loading: false, messages: new_messages)}
  end

  @impl true
  def handle_info(_, socket) do
    {:noreply, socket}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="flex flex-col grow px-2 sm:px-4 lg:px-8 py-10">
      <div class="flex flex-col grow relative -mb-8 mt-2 mt-2">
        <div class="absolute inset-0 gap-4">
          <div class="h-full flex flex-col bg-white shadow-sm border rounded-md">
            <div class="grid-cols-4 h-full grid divide-x">
              <div :if={!Enum.empty?(@books)} class="flex flex-col overflow-scroll overflow-x-hidden hover:scroll-auto">
                <div class="flex flex-col justify-stretch grow p-2">
                  <%= for book <- @books do %>
                  <div id={"doc-#{book.id}"} class="flex flex-col justify-stretch">
                    <button type="button" phx-click="select_book" phx-value-id={book.id} class={"flex p-4 items-center justify-between rounded-md hover:bg-gray-100 text-sm text-left text-gray-700 outline-none #{if @selected && @selected.id == book.id, do: "bg-gray-100"}"}>
                      <div class="flex flex-col overflow-hidden">
                        <div class="inline-flex items-center space-x-1 font-medium text-sm text-gray-800">
                          <div class="p-1 rounded-full bg-gray-200 text-gray-900">
                            <div class="rounded-full w-9 h-9 min-w-9 flex justify-center items-center text-base bg-purple-600 text-white capitalize"><%= String.first(book.title) %></div>
                          </div>
                          <span class="pl-1 capitalize"><%= book.title %></span>
                        </div>
                        <div class="hidden mt-1 inline-flex justify-start items-center flex-nowrap text-xs text-gray-500 overflow-hidden">
                          <span class="whitespace-nowrap text-ellipsis overflow-hidden"><%= book.title %></span>
                          <span class="mx-1 inline-flex rounded-full w-0.5 h-0.5 min-w-0.5 bg-gray-500"></span>
                        </div>
                      </div>
                    </button>
                  </div>
                  <% end %>
                </div>
              </div>
              <div class={"block relative #{if Enum.empty?(@books), do: "col-span-4", else: "col-span-3"}"}>
                <div class="flex absolute inset-0 flex-col">
                  <div class="relative flex grow overflow-y-hidden">
                    <div :if={!is_nil(@selected)} class="pt-4 pb-1 px-4 flex flex-col grow overflow-y-auto">
                      <%= for message <- Enum.filter(@messages, fn m -> m.book_id == @selected.id end) do %>
                      <div :if={message.user_id != 1} class="my-2 flex flex-row justify-start space-x-1 self-start items-start">
                        <div class="flex flex-col space-y-0.5 self-start items-start">
                          <div class="bg-gray-200 text-gray-900 ml-0 mr-12 py-2 px-3 inline-flex text-sm rounded-lg whitespace-pre-wrap"><%= message.text %></div>
                          <div class="mx-1 text-xs text-gray-500"><%= Calendar.strftime(message.inserted_at, "%B %d, %-I:%M %p") %></div>
                        </div>
                      </div>
                      <div :if={message.user_id == 1} class="my-2 flex flex-row justify-start space-x-1 self-end items-end">
                        <div class="flex flex-col space-y-0.5 self-end items-end">
                          <div class="bg-purple-600 text-gray-50 ml-12 mr-0 py-2 px-3 inline-flex text-sm rounded-lg whitespace-pre-wrap"><%= message.text %></div>
                          <div class="mx-1 text-xs text-gray-500"><%= Calendar.strftime(message.inserted_at, "%B %d, %-I:%M %p") %></div>
                        </div>
                      </div>
                      <% end %>
                      <div :if={@loading} class="typing"><div class="typing__dot"></div><div class="typing__dot"></div><div class="typing__dot"></div></div>
                    </div>
                  </div>
                  <form class="px-4 py-2 flex flex-row items-end gap-x-2" phx-submit="add_message" phx-change="change_text">
                    <div id="dragme" class={"flex flex-col grow rounded-md #{if !is_nil(@path), do: "border"} #{if @focused, do: "ring-1 border-indigo-500 ring-indigo-500 border"}"}>
                      <div :if={!is_nil(@path)} class="mx-2 mt-3 mb-2 flex flex-row items-center rounded-md gap-x-4 gap-y-3 flex-wrap">
                        <div class="relative">
                          <div class="px-2 h-14 min-w-14 min-h-14 inline-flex items-center gap-x-2 text-sm rounded-lg whitespace-pre-wrap bg-gray-200 text-gray-900 bg-gray-200 text-gray-900 max-w-24 sm:max-w-32">
                            <div class="p-2 inline-flex justify-center items-center rounded-full bg-gray-300 text-gray-900 bg-gray-300 text-gray-900">
                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" class="w-5 h-5">
                                <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"></path>
                              </svg>
                            </div>
                            <span class="truncate"><%= String.split(@path, "/") |> List.last() %></span>
                          </div>
                          <div :if={@loadingpdf} class="flex p-1 absolute -top-2 -right-2 rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 border border-gray-300 shadow">
                            <div class="text-gray-700 inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent motion-reduce:animate-[spin_1.5s_linear_infinite]" role="status">
                              <span class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Loading...</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div class="relative flex grow">
                        <input id="message" name="message" value={@text} class={"#{if !is_nil(@path), do: "border-transparent"} block w-full rounded-md border-gray-300 shadow-sm #{if is_nil(@path), do: "focus:border-indigo-500 focus:ring-indigo-500"} text-sm placeholder:text-gray-400 text-gray-900"} placeholder="Search ..." type="text" autocomplete="off" spellcheck="false" autocapitalize="off" />
                      </div>
                    </div>
                    <div class="ml-1">
                        <button disabled={is_nil(@path) && !@selected} type="submit" class={"flex items-center justify-center h-10 w-10 rounded-full #{if is_nil(@path) && !@selected, do: "cursor-not-allowed bg-gray-100 text-gray-300", else: "hover:bg-gray-300 bg-gray-200 text-gray-500"}"}>
                        <svg class="w-5 h-5 transform rotate-90 -mr-px" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                        </svg>
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
  end
end
