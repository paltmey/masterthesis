import * as monaco from "monaco-editor/esm/vs/editor/editor.api";

monaco.languages.registerCompletionItemProvider('python', {
    provideCompletionItems: function (model, position) {
        console.log('provideCompletionItems')

        const data = {
            "content": model.getValueInRange(model.getFullModelRange()),
            "position": position
        }

        return fetch('/completion',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json;charset=utf-8',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(tokens => {
                const suggestions = tokens["suggestions"].map(token => {
                    return {
                        label: token["label"],
                        kind: monaco.languages.CompletionItemKind[token["kind"]],
                        documentation: token["documentation"],
                        insertText: token["insertText"],
                    }
                });
                return {
                    suggestions: suggestions
                };
            })
    }
});

//create div to avoid needing a HtmlWebpackPlugin template
const container = document.createElement("div");
container.id = "root";
container.style = "width: 800px; height: 600px; border: 1px solid grey";
document.body.appendChild(container);

monaco.editor.create(document.getElementById("root"), {
    language: 'python',
    theme: "vs-dark"
});
