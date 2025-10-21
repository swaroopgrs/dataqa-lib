# DataQA Development Backlog

## Schema Validation Features

### Foreign Key Validation
- **Priority**: Medium
- **Description**: Add comprehensive validation for foreign key relationships in schema files
- **Requirements**:
  - Validate that foreign key columns exist in source tables
  - Validate that referenced tables exist in schema
  - Validate that referenced columns exist in target tables
  - Support both table-level (`foreign_keys` array) and column-level (`foreign_key` string) formats
  - Provide clear error messages for validation failures
- **Implementation Notes**:
  - Should validate during schema loading in ResourceManager
  - Format validation for column-level foreign keys (expect "table.column" format)
  - Consider performance impact for large schemas
- **Related**: Config format migration spec task 5

### Primary Key Validation
- **Priority**: Low
- **Description**: Validate that primary key references point to existing columns
- **Requirements**:
  - Check that all columns listed in `primary_keys` array exist in the table
  - Provide clear error messages for missing primary key columns
- **Related**: Part of comprehensive schema validation

### Categorical Values Validation
- **Priority**: Low
- **Description**: Add validation for CategoricalValue structures in column definitions
- **Requirements**:
  - Validate that categorical values have proper value-description pair structure
  - Ensure value field is non-empty string
  - Handle optional description field gracefully
  - Provide clear error messages for malformed categorical values
- **Implementation Notes**:
  - Should validate during schema loading in ResourceManager
  - Consider validation of value uniqueness within a column
  - Handle edge cases like empty values list
- **Related**: Config format migration spec task 6

## Future Enhancements

### Schema Relationship Mapping
- **Priority**: Low
- **Description**: Build relationship graph from foreign key definitions
- **Use Cases**:
  - Query optimization
  - Join suggestion
  - Data lineage tracking

### Schema Validation Configuration
- **Priority**: Low
- **Description**: Allow users to configure validation strictness levels
- **Options**:
  - `strict`: Full validation (all relationships must be valid)
  - `format`: Format validation only (current simplified approach)
  - `none`: No validation (trust user input completely)
## Configuration Template Updates

### Template Configuration Migration
- **Priority**: Low
- **Description**: Update template configuration files to new format
- **Requirements**:
  - Update dataqa/templates/schema.yml to new format
  - Update dataqa/templates/rules.yml to new format
  - Update dataqa/templates/examples.yml to new format
  - Ensure templates reflect the new structure for future use
- **Implementation Notes**:
  - Remove metadata sections and flatten structure
  - Convert primary_key to primary_keys array format
  - Convert foreign_key to foreign_keys with proper ForeignKey objects
  - Add enhanced column metadata (row_count, distinct_count, null_count, etc.)
  - Preserve existing business logic and examples
- **Related**: Config format migration spec task 10, Requirements 4.1, 4.2, 4.3
