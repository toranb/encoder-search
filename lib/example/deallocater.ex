defmodule Example.Deallocater do
  def deep_deallocate(data) do
    do_deep_deallocate(data)
  end

  defp do_deep_deallocate(%Nx.Tensor{} = tensor) do
    Nx.backend_deallocate(tensor)
  end

  defp do_deep_deallocate(data) when is_list(data) do
    Enum.each(data, &do_deep_deallocate/1)
  end

  defp do_deep_deallocate(data) when is_tuple(data) do
    data
    |> Tuple.to_list()
    |> Enum.each(&do_deep_deallocate/1)
  end

  defp do_deep_deallocate(data) when is_map(data) do
    data
    |> Map.values()
    |> Enum.each(&do_deep_deallocate/1)
  end

  defp do_deep_deallocate(_other) do
    :ok
  end
end
