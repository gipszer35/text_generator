find . -name '*.xml' -print0 |
while IFS= read -r -d '' file; do
    xmlstarlet sel -t \
        -m '//*[local-name()="body"]//*[local-name()="p"]' \
        -v 'normalize-space(.)' -n \
        "$file"
done
