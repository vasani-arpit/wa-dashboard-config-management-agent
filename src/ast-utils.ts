import { parse } from "@babel/parser";
import _traverse from "@babel/traverse";
import _generate from "@babel/generator";
import * as t from "@babel/types";

// ESM/CJS compatibility handling for Babel imports
const traverse = (_traverse as any).default || _traverse;
const generate = (_generate as any).default || _generate;

/**
 * Inserts or updates a value inside the default export object of a config file.
 */
export function updateConfigField(code: string, key: string, value: any): string {
    const ast = parse(code, {
        sourceType: "module",
        plugins: ["typescript"]
    });

    let updated = false;

    traverse(ast, {
        ExportDefaultDeclaration(path: any) {
            const declaration = path.node.declaration;
            if (t.isObjectExpression(declaration)) {
                const existingPropIndex = declaration.properties.findIndex(
                    (p: any) => t.isObjectProperty(p) && t.isIdentifier(p.key, { name: key })
                );

                let valueNode: t.Expression;
                if (typeof value === "boolean") {
                    valueNode = t.booleanLiteral(value);
                } else if (typeof value === "string") {
                    valueNode = t.stringLiteral(value);
                } else if (Array.isArray(value)) {
                    valueNode = t.arrayExpression(
                        value.map(item => t.stringLiteral(String(item)))
                    );
                } else if (value === null) {
                    valueNode = t.nullLiteral();
                } else {
                    throw new Error(`Unsupported value type: ${typeof value}`);
                }

                if (existingPropIndex !== -1) {
                    const existingProp = declaration.properties[existingPropIndex] as t.ObjectProperty;
                    existingProp.value = valueNode;
                    updated = true;
                } else {
                    const newProp = t.objectProperty(t.identifier(key), valueNode);
                    declaration.properties.push(newProp);
                    updated = true;
                }
            }
        }
    });

    if (!updated) {
        throw new Error("Unable to locate 'export default' object declaration in configuration file.");
    }

    const output = generate(ast, {
        retainLines: true,
        keepComments: true,
    }, code);

    return output.code;
}

/**
 * Extracts comments from file and returns customer name and WhatsApp/Phone structure.
 */
export function parseComments(content: string): { customerName: string | null; phone: string | null } {
    const lines = content.split('\n');
    let customerName: string | null = null;
    let phone: string | null = null;

    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('//')) {
            const comment = trimmed.substring(2).trim();

            const nameMatch = comment.match(/(?:The name customer is|Customer is|Name is)\s+([^\.]+)/i);
            if (nameMatch) {
                customerName = nameMatch[1].trim();
            }

            const phoneMatch = comment.match(/(?:whatsapp number|phone number|whatsapp|phone)\s+(?:in this business\s+)?is\s+(\+?[0-9\s\-]+)/i);
            if (phoneMatch) {
                phone = phoneMatch[1].trim();
            }
        }
    }

    return { customerName, phone };
}

/**
 * Inspects default export object fields to resolve current values.
 */
export function parseFieldValueFromCode(code: string, key: string): any {
    const ast = parse(code, {
        sourceType: "module"
    });

    let value: any = undefined;

    traverse(ast, {
        ExportDefaultDeclaration(path: any) {
            const declaration = path.node.declaration;
            if (t.isObjectExpression(declaration)) {
                const prop = declaration.properties.find(
                    (p: any) => t.isObjectProperty(p) && t.isIdentifier(p.key, { name: key })
                ) as t.ObjectProperty;

                if (prop) {
                    const valNode = prop.value;
                    if (t.isBooleanLiteral(valNode)) {
                        value = valNode.value;
                    } else if (t.isStringLiteral(valNode)) {
                        value = valNode.value;
                    } else if (t.isArrayExpression(valNode)) {
                        value = valNode.elements.map((el: any) => t.isStringLiteral(el) ? el.value : null);
                    } else if (t.isNullLiteral(valNode)) {
                        value = null;
                    }
                }
            }
        }
    });

    return value;
}