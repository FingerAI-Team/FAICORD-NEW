from src import MilvusEnvManager
import argparse
import json
import os 


def main(args):
    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)

    milvus_db = MilvusEnvManager(db_args)
    milvus_db.set_env()
    print(f'client: {milvus_db.client}')
    with open(os.path.join(args.config_path, args.schema_name)) as f:
        schema_config = json.load(f)

    collection_name = schema_config.get('collection_name', 'default_collection')
    schema_fields = [] 
    for field in schema_config['fields']:
        schema_fields.append(
            milvus_db.create_field_schema(
                field['name'],
                dtype=field['dtype'],
                is_primary=field.get('is_primary', False),
                max_length=field.get('max_length'),
                dim=field.get('dim')
            )
        )
    schema = milvus_db.create_schema(schema_fields, schema_config.get("description", "schema"))
    collection = milvus_db.create_collection(collection_name, schema, shards_num=2)
    milvus_db.get_collection_info(collection_name)
    milvus_db.create_index(collection, field_name='audio_emb')   # text 필드에 index 생성 

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--schema_name', type=str, default='schema_config.json')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)