from marshmallow import Schema, fields, ValidationError

class TransactionSchema(Schema):
    features = fields.List(fields.Float(), required=True)

transaction_schema = TransactionSchema()
