defmodule Example.Repo.Migrations.AddVerseTokens do
  use Ecto.Migration

  def change do
    create table(:verse_tokens) do
      add :token_pos, :integer, null: false
      add :embedding, :vector, size: 768

      add :verse_id, references(:verses), null: false

      timestamps()
    end

    create index(:verse_tokens, [:verse_id])
    create index("verse_tokens", ["embedding vector_cosine_ops"], using: :hnsw)

  end
end
