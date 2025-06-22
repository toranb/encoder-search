defmodule Example.Repo.Migrations.AddVerses do
  use Ecto.Migration

  def change do
    create table(:books) do
      add :title, :string, null: false

      timestamps()
    end

    create table(:verses) do
      add :chapter, :integer, null: false
      add :verse, :integer, null: false
      add :text, :text, null: false
      add :embedding, :vector, size: 768

      add :book_id, references(:books), null: false

      timestamps()
    end

    create index(:verses, [:book_id])
    create index("verses", ["embedding vector_cosine_ops"], using: :hnsw)
  end
end
